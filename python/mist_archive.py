"""
mist_archive.py - Python wrapper for mist ASCII archive format

Format specification:
- Scalars: name = value
- Strings: name = "value" (with escape sequences)
- Arrays: name = [v1, v2, v3]
- Groups: name { ... }
- Anonymous groups: { ... } (for vectors of compound types)
- Comments: # to end of line

File extensions:
- .cfg: Configuration files (input)
- .dat: Data files - checkpoints, products, timeseries (output)

Usage:
    import mist_archive as ma

    # Read files
    config = ma.load("config.cfg")
    checkpoint = ma.load("chkpt.0000.dat")

    # Write files
    ma.dump(config, "config.cfg")

    # Parse from string
    data = ma.loads("value = 42")

    # Serialize to string
    text = ma.dumps({"value": 42})
"""

from __future__ import annotations

import numpy as np
from typing import Any, TextIO, Union
from pathlib import Path


class ParseError(Exception):
    """Error during archive parsing."""

    def __init__(self, message: str, line: int = None, context: str = None):
        self.line = line
        self.context = context
        full_msg = message
        if line is not None:
            full_msg = f"Line {line}: {message}"
        if context:
            full_msg += f" (in {context})"
        super().__init__(full_msg)


class AsciiReader:
    """Reader for mist ASCII archive format."""

    def __init__(self, source: Union[str, Path, TextIO]):
        if isinstance(source, (str, Path)):
            self._file = open(source, "r")
            self._owns_file = True
        else:
            self._file = source
            self._owns_file = False
        self._line_num = 1
        self._group_stack: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._owns_file:
            self._file.close()

    @property
    def _context(self) -> str:
        return "/".join(self._group_stack) if self._group_stack else "root"

    def _skip_whitespace_and_comments(self):
        """Skip whitespace and # comments."""
        while True:
            c = self._peek()
            if c == "":
                return
            if c in " \t\r":
                self._get()
            elif c == "\n":
                self._get()
                self._line_num += 1
            elif c == "#":
                # Skip to end of line
                while self._peek() not in ("", "\n"):
                    self._get()
            else:
                return

    def _peek(self) -> str:
        pos = self._file.tell()
        c = self._file.read(1)
        self._file.seek(pos)
        return c

    def _get(self) -> str:
        return self._file.read(1)

    def _expect(self, expected: str):
        c = self._get()
        if c != expected:
            raise ParseError(
                f"Expected '{expected}', got '{c}'", self._line_num, self._context
            )

    def _read_identifier(self) -> str:
        """Read alphanumeric identifier (a-z, A-Z, 0-9, _)."""
        result = []
        while True:
            c = self._peek()
            if c.isalnum() or c == "_":
                result.append(self._get())
            else:
                break
        return "".join(result)

    def _read_number(self) -> Union[int, float]:
        """Read a numeric value."""
        result = []
        is_float = False

        # Handle negative
        if self._peek() == "-":
            result.append(self._get())

        while True:
            c = self._peek()
            if c.isdigit():
                result.append(self._get())
            elif c in ".eE":
                is_float = True
                result.append(self._get())
            elif c in "+-":
                # Only valid after e/E
                if result and result[-1] in "eE":
                    result.append(self._get())
                else:
                    break
            else:
                break

        value_str = "".join(result)
        try:
            if is_float:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            raise ParseError(
                f"Invalid number: {value_str}", self._line_num, self._context
            )

    def _read_quoted_string(self) -> str:
        """Read a quoted string with escape sequences."""
        self._expect('"')
        result = []

        while True:
            c = self._get()
            if c == "":
                raise ParseError("Unterminated string", self._line_num, self._context)
            if c == '"':
                break
            if c == "\\":
                escaped = self._get()
                escape_map = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
                result.append(escape_map.get(escaped, escaped))
            else:
                result.append(c)

        return "".join(result)

    def _read_array(self) -> np.ndarray:
        """Read an array [v1, v2, ...]."""
        self._expect("[")
        values = []

        self._skip_whitespace_and_comments()

        if self._peek() == "]":
            self._get()
            return np.array([])

        while True:
            self._skip_whitespace_and_comments()
            values.append(self._read_number())
            self._skip_whitespace_and_comments()

            c = self._peek()
            if c == "]":
                self._get()
                break
            elif c == ",":
                self._get()
            else:
                raise ParseError(
                    f"Expected ',' or ']', got '{c}'", self._line_num, self._context
                )

        return np.array(values)

    def _read_value(self) -> Any:
        """Read a value (number, string, or array)."""
        self._skip_whitespace_and_comments()
        c = self._peek()

        if c == '"':
            return self._read_quoted_string()
        elif c == "[":
            return self._read_array()
        else:
            return self._read_number()

    def read_all(self) -> dict:
        """Read entire archive into nested dict."""
        result = {}

        while True:
            self._skip_whitespace_and_comments()
            if self._peek() == "":
                break

            name = self._read_identifier()
            if not name:
                break

            self._skip_whitespace_and_comments()
            c = self._peek()

            if c == "=":
                # Scalar or array assignment
                self._get()
                self._skip_whitespace_and_comments()
                result[name] = self._read_value()
            elif c == "{":
                # Named group
                self._get()
                self._group_stack.append(name)
                result[name] = self._read_group_contents()
                self._group_stack.pop()
            else:
                raise ParseError(
                    f"Expected '=' or '{{', got '{c}'", self._line_num, self._context
                )

        return result

    def _read_group_contents(self) -> Union[dict, list]:
        """Read contents of a group (may be dict or list of anonymous groups)."""
        self._skip_whitespace_and_comments()

        # Check if this is a list of anonymous groups
        if self._peek() == "{":
            # Anonymous groups -> return list
            items = []
            while True:
                self._skip_whitespace_and_comments()
                if self._peek() == "}":
                    self._get()
                    return items
                if self._peek() == "{":
                    self._get()
                    items.append(self._read_group_contents())
                else:
                    raise ParseError(
                        "Expected '{' or '}'", self._line_num, self._context
                    )

        # Named fields -> return dict
        result = {}
        while True:
            self._skip_whitespace_and_comments()

            if self._peek() == "}":
                self._get()
                return result

            name = self._read_identifier()
            if not name:
                raise ParseError(
                    "Expected field name or '}'", self._line_num, self._context
                )

            self._skip_whitespace_and_comments()
            c = self._peek()

            if c == "=":
                self._get()
                self._skip_whitespace_and_comments()
                result[name] = self._read_value()
            elif c == "{":
                self._get()
                self._group_stack.append(name)
                result[name] = self._read_group_contents()
                self._group_stack.pop()
            else:
                raise ParseError(
                    f"Expected '=' or '{{', got '{c}'", self._line_num, self._context
                )

        return result


class AsciiWriter:
    """Writer for mist ASCII archive format."""

    def __init__(self, dest: Union[str, Path, TextIO], indent_size: int = 4):
        if isinstance(dest, (str, Path)):
            self._file = open(dest, "w")
            self._owns_file = True
        else:
            self._file = dest
            self._owns_file = False
        self._indent_size = indent_size
        self._indent_level = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._owns_file:
            self._file.close()

    def _indent(self) -> str:
        return " " * (self._indent_level * self._indent_size)

    def _format_number(self, value: Union[int, float, np.integer, np.floating]) -> str:
        if isinstance(value, (float, np.floating)):
            s = f"{value:.15g}"
            # Ensure decimal point or scientific notation for floats
            if "." not in s and "e" not in s and "E" not in s:
                s += ".0"
            return s
        return str(int(value))

    def _format_string(self, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        return f'"{escaped}"'

    def _format_array(self, arr: np.ndarray) -> str:
        values = ", ".join(self._format_number(v) for v in arr.flat)
        return f"[{values}]"

    def write_all(self, data: dict):
        """Write entire nested dict to archive."""
        for name, value in data.items():
            self._write_field(name, value)

    def _write_field(self, name: str, value: Any):
        if isinstance(value, dict):
            self._file.write(f"{self._indent()}{name} {{\n")
            self._indent_level += 1
            for k, v in value.items():
                self._write_field(k, v)
            self._indent_level -= 1
            self._file.write(f"{self._indent()}}}\n")
        elif isinstance(value, list):
            # List of compound objects (anonymous groups)
            self._file.write(f"{self._indent()}{name} {{\n")
            self._indent_level += 1
            for item in value:
                self._file.write(f"{self._indent()}{{\n")
                self._indent_level += 1
                for k, v in item.items():
                    self._write_field(k, v)
                self._indent_level -= 1
                self._file.write(f"{self._indent()}}}\n")
            self._indent_level -= 1
            self._file.write(f"{self._indent()}}}\n")
        elif isinstance(value, np.ndarray):
            self._file.write(f"{self._indent()}{name} = {self._format_array(value)}\n")
        elif isinstance(value, str):
            self._file.write(f"{self._indent()}{name} = {self._format_string(value)}\n")
        elif isinstance(value, (int, float, np.integer, np.floating)):
            self._file.write(f"{self._indent()}{name} = {self._format_number(value)}\n")
        else:
            raise TypeError(f"Unsupported type: {type(value)}")


# Convenience functions


def load(source: Union[str, Path, TextIO]) -> dict:
    """Load an ASCII archive file into a nested dictionary.

    Args:
        source: File path (str or Path) or file-like object

    Returns:
        Nested dictionary with archive contents

    Example:
        config = ma.load("config.cfg")
        checkpoint = ma.load("chkpt.0000.dat")
    """
    with AsciiReader(source) as reader:
        return reader.read_all()


def dump(data: dict, dest: Union[str, Path, TextIO], indent_size: int = 4):
    """Write a nested dictionary to an ASCII archive file.

    Args:
        data: Nested dictionary to serialize
        dest: File path (str or Path) or file-like object
        indent_size: Number of spaces per indentation level (default: 4)

    Example:
        ma.dump(config, "config.cfg")
        ma.dump(checkpoint, "chkpt.0001.dat")
    """
    with AsciiWriter(dest, indent_size) as writer:
        writer.write_all(data)


def loads(text: str) -> dict:
    """Parse ASCII archive from string.

    Args:
        text: Archive content as string

    Returns:
        Nested dictionary with archive contents

    Example:
        data = ma.loads("value = 42")
    """
    import io

    return load(io.StringIO(text))


def dumps(data: dict, indent_size: int = 4) -> str:
    """Serialize to ASCII archive string.

    Args:
        data: Nested dictionary to serialize
        indent_size: Number of spaces per indentation level (default: 4)

    Returns:
        Archive content as string

    Example:
        text = ma.dumps({"value": 42})
    """
    import io

    buf = io.StringIO()
    dump(data, buf, indent_size)
    return buf.getvalue()
