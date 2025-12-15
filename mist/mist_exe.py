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
        self.sock = None
        self.port = None
        # Make stdout non-blocking
        os.set_blocking(self.process.stdout.fileno(), False)

    def send_command(self, cmd: str) -> str:
        """Send a command to the mist process and return available stdout."""
        self.process.stdin.write((cmd + "\n").encode())
        self.process.stdin.flush()

        # Wait a bit then read whatever is available
        output = self._read_available(timeout=0.2)

        # Check for socket open - extract port and connect
        if "socket listening on port" in output:
            match = re.search(r"port (\d+)", output)
            if match:
                self.port = int(match.group(1))
                self._connect_socket()
                # Read the "connected" response
                output += self._read_available(timeout=0.1)

        return output

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

    def _connect_socket(self):
        """Connect to the mist socket server."""
        if self.port is None:
            raise RuntimeError("No socket port available")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", self.port))

    def read_socket(self) -> dict:
        """Read size-prefixed binary data from socket and parse as mist archive."""
        if self.sock is None:
            raise RuntimeError("Socket not connected; use 'socket open' first")

        # Read size prefix (uint64)
        size_data = self._recv_exact(8)
        size = struct.unpack("<Q", size_data)[0]

        # Read data
        data = self._recv_exact(size)

        # Parse with BinaryReader
        reader = mist_archive.BinaryReader(io.BytesIO(data))
        return reader.read_all()

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(min(65536, n - len(data)))
            if not chunk:
                raise RuntimeError("Socket connection closed")
            data += chunk
        return data

    def close(self):
        """Close socket and terminate process."""
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
