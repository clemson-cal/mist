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

def _serialize_command(cmd_name: str, fields: dict) -> bytes:
    """Serialize a command to binary format for socket transmission."""
    if cmd_name not in COMMAND_INDICES:
        raise ValueError(f"Unknown command: {cmd_name}")

    # Transform fields: convert 'dest' to optional format
    transformed = {}
    for name, value in fields.items():
        if name == "dest" and isinstance(value, str):
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
# High-level interface
# =============================================================================

class Mist:
    """High-level interface for running mist simulations.

    Example:
        sim = Mist("./advect1d")
        sim.init()
        sim.select_products(["concentration"])
        sim.run(t=1.0)
        print(sim.products)
        sim.close()
    """

    def __init__(self, executable: str):
        """Connect to a mist executable.

        Args:
            executable: Path to the mist executable
        """
        self._conn = _MistConnection(executable)
        self._initialized = False

    def _check_response(self, responses: list[tuple[str, dict]]):
        """Check responses for errors."""
        for resp_type, resp_data in responses:
            if resp_type == "error":
                raise RuntimeError(resp_data.get("what", "Unknown error"))

    # --- Lifecycle ---

    def init(self):
        """Initialize the simulation state."""
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

    def run(self, *, t: Optional[float] = None, dt: Optional[float] = None, var: str = "t"):
        """Advance the simulation.

        Args:
            t: Target time to advance to
            dt: Time increment to advance by
            var: Time variable name (default: "t")

        Exactly one of t or dt must be specified.
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call init() first.")
        if (t is None) == (dt is None):
            raise ValueError("Specify exactly one of t= or dt=")

        if t is not None:
            responses = self._conn.send_command("advance_to", {"var": var, "target": t})
        else:
            responses = self._conn.send_command("advance_by", {"var": var, "delta": dt})
        self._check_response(responses)

    # --- Configuration ---

    def set_physics(self, key: str, value: str):
        """Set a physics configuration parameter."""
        responses = self._conn.send_command("set_physics", {"key": key, "value": value})
        self._check_response(responses)

    def set_initial(self, key: str, value: str):
        """Set an initial configuration parameter."""
        responses = self._conn.send_command("set_initial", {"key": key, "value": value})
        self._check_response(responses)

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
    def products(self) -> dict:
        """Get current products data."""
        return self._conn.fetch_data("write_products")

    @property
    def checkpoint(self) -> dict:
        """Get checkpoint (full state) data."""
        return self._conn.fetch_data("write_checkpoint")

    @property
    def timeseries(self) -> dict:
        """Get recorded timeseries data."""
        return self._conn.fetch_data("write_timeseries")

    @property
    def physics(self) -> dict:
        """Get physics configuration."""
        return self._conn.fetch_data("write_physics")

    @property
    def initial(self) -> dict:
        """Get initial configuration."""
        return self._conn.fetch_data("write_initial")

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
        """Get current simulation time (shortcut for state['times'])."""
        state = self.state
        times = state.get("times", [])
        for entry in times:
            if entry.get("key") == "t":
                return entry.get("value", 0.0)
        return 0.0
