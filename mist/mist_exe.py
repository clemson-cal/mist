"""Interface for running mist executables and communicating via socket."""

import io
import os
import re
import select
import socket
import struct
import subprocess

from . import mist_archive


class Mist:
    """Interface for running mist simulations and reading data via socket."""

    def __init__(self, executable: str):
        self.executable = executable
        self.process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Make stdout non-blocking
        os.set_blocking(self.process.stdout.fileno(), False)

    def send(self, cmd: str) -> str:
        """Send a command to the mist process and return available stdout."""
        self.process.stdin.write((cmd + "\n").encode())
        self.process.stdin.flush()
        return self._read_available(timeout=0.2)

    def _read_available(self, timeout: float = 0.1) -> str:
        """Read whatever output is available within timeout."""
        result = []
        while True:
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not ready:
                break
            chunk = self.process.stdout.read(4096)
            if not chunk:
                break
            result.append(chunk.decode())
            timeout = 0.01  # Shorter timeout for subsequent reads
        return "".join(result)

    def _write_to_socket(self, cmd: str) -> dict:
        """Send a write command and receive data via socket.

        The driver opens a socket, prints the port, waits for connection,
        sends data, then closes. This method handles the full transaction.
        """
        self.process.stdin.write((cmd + "\n").encode())
        self.process.stdin.flush()

        # Read output to get the port
        output = self._read_available(timeout=0.5)
        match = re.search(r"socket listening on port (\d+)", output)
        if not match:
            raise RuntimeError(f"Expected socket port in output: {output}")

        port = int(match.group(1))

        # Connect to the socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(("127.0.0.1", port))

            # Read size prefix (uint64)
            size_data = self._recv_exact(sock, 8)
            size = struct.unpack("<Q", size_data)[0]

            # Read data
            data = self._recv_exact(sock, size)
        finally:
            sock.close()

        # Read remaining output (sent bytes message)
        self._read_available(timeout=0.1)

        # Parse with BinaryReader
        reader = mist_archive.BinaryReader(io.BytesIO(data))
        return reader.read_all()

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = sock.recv(min(65536, n - len(data)))
            if not chunk:
                raise RuntimeError("Socket connection closed")
            data += chunk
        return data

    def get_products(self) -> dict:
        """Get current products from simulation."""
        return self._write_to_socket("write products socket")

    def get_timeseries(self) -> dict:
        """Get timeseries data from simulation."""
        return self._write_to_socket("write timeseries socket")

    def get_checkpoint(self) -> dict:
        """Get full checkpoint (state) from simulation."""
        return self._write_to_socket("write checkpoint socket")

    def get_physics(self) -> dict:
        """Get physics configuration."""
        return self._write_to_socket("write physics socket")

    def get_initial(self) -> dict:
        """Get initial configuration."""
        return self._write_to_socket("write initial socket")

    def get_driver(self) -> dict:
        """Get driver state."""
        return self._write_to_socket("write driver socket")

    def get_profiler(self) -> dict:
        """Get profiler data."""
        return self._write_to_socket("write profiler socket")

    def close(self):
        """Terminate the process."""
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
