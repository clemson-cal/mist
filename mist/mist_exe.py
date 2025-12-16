"""Interface for running mist executables and communicating via socket."""

import io
import socket
import struct
import subprocess
from typing import Any, Optional

from . import mist_archive


# =============================================================================
# Command/Response definitions - mirrors C++ variants
# =============================================================================

COMMAND_INDICES = {
    "advance_by": 0,
    "advance_to": 1,
    "set_output": 2,
    "set_physics": 3,
    "set_initial": 4,
    "set_exec": 5,
    "select_timeseries": 6,
    "select_products": 7,
    "do_timeseries": 8,
    "write_physics": 9,
    "write_initial": 10,
    "write_driver": 11,
    "write_profiler": 12,
    "write_timeseries": 13,
    "write_checkpoint": 14,
    "write_products": 15,
    "write_iteration": 16,
    "clear_repeat": 17,
    "init": 18,
    "reset": 19,
    "load": 20,
    "show_state": 21,
    "show_all": 22,
    "show_physics": 23,
    "show_initial": 24,
    "show_iteration": 25,
    "show_timeseries": 26,
    "show_products": 27,
    "show_profiler": 28,
    "show_driver": 29,
    "help": 30,
    "help_schema": 31,
    "stop": 32,
}

RESPONSE_NAMES = [
    "ok", "error", "interrupted", "stopped", "state_info",
    "iteration_info", "timeseries_sample", "physics_config",
    "initial_config", "driver_state", "help_text", "timeseries_info",
    "products_info", "profiler_info", "wrote_file", "socket_listening",
    "socket_sent", "socket_cancelled",
]

TERMINAL_RESPONSES = {
    "ok", "error", "interrupted", "stopped",
    "socket_sent", "socket_cancelled", "wrote_file",
    "state_info", "iteration_info", "timeseries_sample",
    "physics_config", "initial_config", "driver_state",
    "help_text", "timeseries_info", "products_info", "profiler_info",
}


# =============================================================================
# Command serialization (handles optional fields for C++ compatibility)
# =============================================================================

# Commands where 'dest' is std::optional<std::string>
COMMANDS_WITH_OPTIONAL_DEST = {
    "write_physics",
    "write_initial",
    "write_driver",
    "write_profiler",
    "write_timeseries",
    "write_checkpoint",
    "write_products",
    "write_iteration",
}


def _serialize_command(cmd_name: str, fields: dict) -> bytes:
    """Serialize a command to binary format for socket transmission."""
    if cmd_name not in COMMAND_INDICES:
        raise ValueError(f"Unknown command: {cmd_name}")

    # Transform fields: convert 'dest' to optional format only for commands that use optional
    transformed = {}
    for name, value in fields.items():
        if name == "dest" and isinstance(value, str) and cmd_name in COMMANDS_WITH_OPTIONAL_DEST:
            # C++ expects std::optional<std::string> as GROUP{has_value, value}
            transformed[name] = {"has_value": 1, "value": value}
        else:
            transformed[name] = value

    buf = io.BytesIO()
    writer = mist_archive.BinaryWriter(buf)
    writer.write_variant(COMMAND_INDICES[cmd_name], transformed)
    return buf.getvalue()


def _deserialize_response(data: bytes) -> tuple[str, dict]:
    """Deserialize a response from binary format."""
    reader = mist_archive.BinaryReader(io.BytesIO(data))
    index, value = reader.read_variant()
    if index < 0 or index >= len(RESPONSE_NAMES):
        raise ValueError(f"Unknown response index: {index}")
    return RESPONSE_NAMES[index], value


# =============================================================================
# Low-level socket connection
# =============================================================================

class _MistConnection:
    """Low-level socket connection to a mist executable."""

    def __init__(self, executable: str):
        self.process = subprocess.Popen(
            [executable, "--socket"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        ports = {}
        for _ in range(3):
            line = self.process.stdout.readline().decode().strip()
            if "=" in line:
                key, value = line.split("=", 1)
                ports[key] = int(value)

        if "command_port" not in ports or "response_port" not in ports:
            self.close()
            raise RuntimeError(f"Failed to get socket ports: {ports}")

        self.data_port = ports.get("data_port")
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.command_socket.connect(("127.0.0.1", ports["command_port"]))
        self.response_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.response_socket.connect(("127.0.0.1", ports["response_port"]))

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(min(65536, n - len(data)))
            if not chunk:
                raise RuntimeError("Socket connection closed")
            data += chunk
        return data

    def send_command(self, cmd_name: str, fields: Optional[dict] = None) -> list[tuple[str, dict]]:
        data = _serialize_command(cmd_name, fields or {})
        self.command_socket.sendall(struct.pack("<Q", len(data)))
        self.command_socket.sendall(data)

        responses = []
        while True:
            size_data = self._recv_exact(self.response_socket, 8)
            size = struct.unpack("<Q", size_data)[0]
            resp_data = self._recv_exact(self.response_socket, size)
            resp_type, resp_fields = _deserialize_response(resp_data)
            responses.append((resp_type, resp_fields))
            if resp_type in TERMINAL_RESPONSES:
                break
        return responses

    def fetch_data(self, cmd_name: str) -> dict:
        """Send write command and fetch data via data socket."""
        data = _serialize_command(cmd_name, {"dest": "socket"})
        self.command_socket.sendall(struct.pack("<Q", len(data)))
        self.command_socket.sendall(data)

        # Wait for socket_listening response
        size_data = self._recv_exact(self.response_socket, 8)
        size = struct.unpack("<Q", size_data)[0]
        resp_data = self._recv_exact(self.response_socket, size)
        resp_type, resp_fields = _deserialize_response(resp_data)

        if resp_type == "error":
            raise RuntimeError(resp_fields.get("what", "Unknown error"))
        if resp_type != "socket_listening":
            raise RuntimeError(f"Expected socket_listening, got {resp_type}")

        port = resp_fields.get("port")
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            data_sock.connect(("127.0.0.1", port))
            size_data = self._recv_exact(data_sock, 8)
            size = struct.unpack("<Q", size_data)[0]
            binary_data = self._recv_exact(data_sock, size)
        finally:
            data_sock.close()

        # Read final response (socket_sent)
        size_data = self._recv_exact(self.response_socket, 8)
        size = struct.unpack("<Q", size_data)[0]
        self._recv_exact(self.response_socket, size)

        return mist_archive.BinaryReader(io.BytesIO(binary_data)).read_all()

    def close(self):
        for sock in [self.command_socket, self.response_socket]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None


# =============================================================================
# Config wrapper classes for dict-like access
# =============================================================================

class _PhysicsConfig:
    """Dict-like wrapper for physics configuration with read/write access."""

    def __init__(self, mist: "Mist"):
        self._mist = mist

    def __getitem__(self, key: str) -> Any:
        data = self._mist._conn.fetch_data("write_physics")
        return data.get("physics", {}).get(key)

    def __setitem__(self, key: str, value: Any):
        self._mist._conn.send_command("set_physics", {"key": key, "value": str(value)})

    def __repr__(self) -> str:
        data = self._mist._conn.fetch_data("write_physics")
        return repr(data.get("physics", {}))

    def keys(self):
        data = self._mist._conn.fetch_data("write_physics")
        return data.get("physics", {}).keys()

    def items(self):
        data = self._mist._conn.fetch_data("write_physics")
        return data.get("physics", {}).items()

    def to_dict(self) -> dict:
        data = self._mist._conn.fetch_data("write_physics")
        return data.get("physics", {})


class _InitialConfig:
    """Dict-like wrapper for initial configuration (read-only after init)."""

    def __init__(self, mist: "Mist"):
        self._mist = mist

    def __getitem__(self, key: str) -> Any:
        data = self._mist._conn.fetch_data("write_initial")
        return data.get("initial", {}).get(key)

    def __setitem__(self, key: str, value: Any):
        if self._mist._initialized:
            raise RuntimeError("Cannot modify initial config after init(). Call reset() first.")
        self._mist._conn.send_command("set_initial", {"key": key, "value": str(value)})

    def __repr__(self) -> str:
        data = self._mist._conn.fetch_data("write_initial")
        return repr(data.get("initial", {}))

    def keys(self):
        data = self._mist._conn.fetch_data("write_initial")
        return data.get("initial", {}).keys()

    def items(self):
        data = self._mist._conn.fetch_data("write_initial")
        return data.get("initial", {}).items()

    def to_dict(self) -> dict:
        data = self._mist._conn.fetch_data("write_initial")
        return data.get("initial", {})


class _Products:
    """Dict-like wrapper for products with array extraction."""

    def __init__(self, mist: "Mist"):
        self._mist = mist

    def __getitem__(self, key: str) -> Any:
        data = self._mist._conn.fetch_data("write_products")
        product = data.get(key)
        if product is None:
            raise KeyError(f"Product '{key}' not found. Available: {list(data.keys())}")
        # If it's a list of patches, concatenate the data arrays
        if isinstance(product, list) and len(product) > 0 and "data" in product[0]:
            import numpy as np
            return np.concatenate([p["data"] for p in product])
        return product

    def __repr__(self) -> str:
        data = self._mist._conn.fetch_data("write_products")
        return f"Products({list(data.keys())})"

    def keys(self):
        data = self._mist._conn.fetch_data("write_products")
        return data.keys()

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def raw(self) -> dict:
        """Get raw products data without concatenation."""
        return self._mist._conn.fetch_data("write_products")


# =============================================================================
# High-level interface
# =============================================================================

class Mist:
    """High-level interface for running mist simulations.

    Example:
        from mist import Mist

        with Mist("./advect1d") as sim:
            u = sim.products["concentration"]
            sim.physics["cfl"] = 0.8
            sim.advance_to(1.0)
            print(sim.time, sim.iteration)

        # With configuration:
        with Mist("./advect1d", physics={"cfl": 0.8}, initial={"num_zones": 400}) as sim:
            sim.advance_by(0.1)
    """

    def __init__(
        self,
        executable: str,
        *,
        physics: Optional[dict] = None,
        initial: Optional[dict] = None,
        exec: Optional[dict] = None,
        init: bool = True,
    ):
        """Connect to a mist executable and optionally initialize.

        Args:
            executable: Path to the mist executable
            physics: Optional physics configuration dict
            initial: Optional initial configuration dict
            exec: Optional execution configuration dict
            init: Whether to auto-initialize (default: True)
        """
        self._conn = _MistConnection(executable)
        self._initialized = False

        # Wrapper objects for dict-like access
        self._physics = _PhysicsConfig(self)
        self._initial = _InitialConfig(self)
        self._products = _Products(self)

        # Apply configuration before init
        if physics:
            for key, value in physics.items():
                self._conn.send_command("set_physics", {"key": key, "value": str(value)})

        if initial:
            for key, value in initial.items():
                self._conn.send_command("set_initial", {"key": key, "value": str(value)})

        if exec:
            for key, value in exec.items():
                self._conn.send_command("set_exec", {"key": key, "value": str(value)})

        if init:
            self.init()

    def _check_response(self, responses: list[tuple[str, dict]]):
        """Check responses for errors."""
        for resp_type, resp_data in responses:
            if resp_type == "error":
                raise RuntimeError(resp_data.get("what", "Unknown error"))

    # --- Lifecycle ---

    def init(self):
        """Initialize the simulation state."""
        if self._initialized:
            return
        responses = self._conn.send_command("init")
        self._check_response(responses)
        self._initialized = True

    def reset(self):
        """Reset the simulation state."""
        responses = self._conn.send_command("reset")
        self._check_response(responses)
        self._initialized = False

    def close(self):
        """Close the connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --- Time evolution ---

    def advance_to(self, target: float, var: str = "t"):
        """Advance simulation to a target value.

        Args:
            target: Target value to advance to
            var: Variable name (default: "t" for time)
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call init() first.")
        responses = self._conn.send_command("advance_to", {"var": var, "target": target})
        self._check_response(responses)

    def advance_by(self, delta: float, var: str = "t"):
        """Advance simulation by a delta.

        Args:
            delta: Amount to advance by
            var: Variable name (default: "t" for time)
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call init() first.")
        responses = self._conn.send_command("advance_by", {"var": var, "delta": delta})
        self._check_response(responses)

    def run(self, *, t: Optional[float] = None, dt: Optional[float] = None, var: str = "t"):
        """Advance the simulation (legacy interface).

        Args:
            t: Target time to advance to
            dt: Time increment to advance by
            var: Time variable name (default: "t")

        Exactly one of t or dt must be specified.
        """
        if (t is None) == (dt is None):
            raise ValueError("Specify exactly one of t= or dt=")

        if t is not None:
            self.advance_to(t, var)
        else:
            self.advance_by(dt, var)

    # --- Configuration (dict-like access) ---

    @property
    def physics(self) -> _PhysicsConfig:
        """Physics configuration (read/write dict-like access)."""
        return self._physics

    @property
    def initial(self) -> _InitialConfig:
        """Initial configuration (read-only after init, dict-like access)."""
        return self._initial

    @property
    def products(self) -> _Products:
        """Products data (dict-like access with automatic array concatenation)."""
        return self._products

    # --- Selection ---

    def select_products(self, names: list[str]):
        """Select products to include when reading products."""
        responses = self._conn.send_command("select_products", {"prods": names})
        self._check_response(responses)

    def select_timeseries(self, names: list[str]):
        """Select timeseries columns to record."""
        responses = self._conn.send_command("select_timeseries", {"cols": names})
        self._check_response(responses)

    # --- Data properties ---

    @property
    def checkpoint(self) -> dict:
        """Get checkpoint (full state) data."""
        return self._conn.fetch_data("write_checkpoint")

    @property
    def timeseries(self) -> dict:
        """Get recorded timeseries data."""
        return self._conn.fetch_data("write_timeseries")

    @property
    def profiler(self) -> dict:
        """Get profiler data."""
        return self._conn.fetch_data("write_profiler")

    @property
    def state(self) -> dict:
        """Get current state info (initialized, zone_count, times)."""
        responses = self._conn.send_command("show_state")
        for resp_type, resp_data in responses:
            if resp_type == "state_info":
                return resp_data
        return {}

    @property
    def time(self) -> float:
        """Get current simulation time."""
        state = self.state
        times = state.get("times", [])
        for entry in times:
            if entry.get("key") == "t":
                return entry.get("value", 0.0)
        return 0.0

    @property
    def iteration(self) -> int:
        """Get current iteration number."""
        responses = self._conn.send_command("show_iteration")
        for resp_type, resp_data in responses:
            if resp_type == "iteration_info":
                return resp_data.get("n", 0)
        return 0

    @property
    def dt(self) -> float:
        """Get last timestep size."""
        responses = self._conn.send_command("show_iteration")
        for resp_type, resp_data in responses:
            if resp_type == "iteration_info":
                return resp_data.get("dt", 0.0)
        return 0.0

    @property
    def zps(self) -> float:
        """Get zones per second."""
        responses = self._conn.send_command("show_iteration")
        for resp_type, resp_data in responses:
            if resp_type == "iteration_info":
                return resp_data.get("zps", 0.0)
        return 0.0

    @property
    def product_names(self) -> list[str]:
        """Get list of available product names."""
        responses = self._conn.send_command("show_products")
        for resp_type, resp_data in responses:
            if resp_type == "products_info":
                return resp_data.get("available", [])
        return []

    @property
    def timeseries_names(self) -> list[str]:
        """Get list of available timeseries column names."""
        responses = self._conn.send_command("show_timeseries")
        for resp_type, resp_data in responses:
            if resp_type == "timeseries_info":
                return resp_data.get("available", [])
        return []
