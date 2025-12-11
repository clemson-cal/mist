"""
mist_archive.py - Python wrapper for mist ASCII and binary archive formats

ASCII Format specification:
- Scalars: name = value
- Strings: name = "value" (with escape sequences)
- Arrays: name = [v1, v2, v3]
- Groups: name { ... }
- Anonymous groups: { ... } (for vectors of compound types)
- Comments: # to end of line

Binary Format specification:
- Scalars: raw bytes in native endianness (8 bytes for double, 4 for int)
- Strings: uint64 length prefix + UTF-8 bytes (no null terminator)
- Arrays: uint64 count prefix + packed elements
- Groups: uint64 child count prefix + serialized children
- Field names are NOT stored (relies on consistent ordering)

File extensions:
- .cfg: Configuration files (ASCII, input)
- .dat: Data files - checkpoints, products, timeseries (ASCII, output)
- .bin: Binary data files (binary format)

Usage:
    import mist_archive as ma

    # Read files (format auto-detected by extension)
    config = ma.load("config.cfg")
    checkpoint = ma.load("chkpt.0000.dat")
    binary_data = ma.load("chkpt.0000.bin")

    # Write files (format auto-detected by extension)
    ma.dump(config, "config.cfg")
    ma.dump(data, "output.bin")  # Binary format

    # Parse from string (ASCII only)
    data = ma.loads("value = 42")

    # Serialize to string (ASCII only)
    text = ma.dumps({"value": 42})

    # Explicit binary operations
    data = ma.load_binary("data.bin")
    ma.dump_binary(data, "data.bin")
"""

from __future__ import annotations

import struct
import numpy as np
from typing import Any, TextIO, BinaryIO, Union
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
        """Read contents of a group (may be dict, list of anonymous groups, or list of strings)."""
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

        # Check if this is a list of bare quoted strings (std::vector<std::string>)
        if self._peek() == '"':
            items = []
            while True:
                self._skip_whitespace_and_comments()
                if self._peek() == "}":
                    self._get()
                    return items
                if self._peek() == '"':
                    items.append(self._read_quoted_string())
                else:
                    raise ParseError(
                        "Expected '\"' or '}'", self._line_num, self._context
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
            if len(value) == 0:
                # Empty list
                self._file.write(f"{self._indent()}{name} {{\n")
                self._file.write(f"{self._indent()}}}\n")
            elif isinstance(value[0], str):
                # List of strings (bare quoted strings in a group)
                self._file.write(f"{self._indent()}{name} {{\n")
                self._indent_level += 1
                for item in value:
                    self._file.write(f"{self._indent()}{self._format_string(item)}\n")
                self._indent_level -= 1
                self._file.write(f"{self._indent()}}}\n")
            elif isinstance(value[0], dict):
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
            else:
                raise TypeError(f"Unsupported list item type: {type(value[0])}")
        elif isinstance(value, np.ndarray):
            self._file.write(f"{self._indent()}{name} = {self._format_array(value)}\n")
        elif isinstance(value, str):
            self._file.write(f"{self._indent()}{name} = {self._format_string(value)}\n")
        elif isinstance(value, (int, float, np.integer, np.floating)):
            self._file.write(f"{self._indent()}{name} = {self._format_number(value)}\n")
        else:
            raise TypeError(f"Unsupported type: {type(value)}")


# =============================================================================
# Binary Format Constants and Reader/Writer (Self-Describing Format)
# =============================================================================

# Magic header and version
BINARY_MAGIC = 0x4D495354  # "MIST" in ASCII
BINARY_VERSION = 1

# Type tags
TYPE_INT32 = 0x01
TYPE_INT64 = 0x02
TYPE_FLOAT64 = 0x03
TYPE_STRING = 0x04
TYPE_ARRAY = 0x05
TYPE_GROUP = 0x06
TYPE_LIST = 0x07

# Element type tags for arrays
ELEM_INT32 = 0x01
ELEM_INT64 = 0x02
ELEM_FLOAT64 = 0x03


class BinaryReader:
    """Reader for mist self-describing binary archive format.

    This format includes field names and type tags, allowing schema-free
    deserialization (like JSON/ASCII format but more compact).
    """

    def __init__(self, source: Union[str, Path, BinaryIO]):
        if isinstance(source, (str, Path)):
            self._file = open(source, "rb")
            self._owns_file = True
        else:
            self._file = source
            self._owns_file = False
        self._header_read = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._owns_file:
            self._file.close()

    def _ensure_header(self):
        if not self._header_read:
            magic = self._read_uint32()
            if magic != BINARY_MAGIC:
                raise ParseError(f"Invalid binary archive: bad magic number 0x{magic:08X}")
            version = self._read_uint8()
            if version != BINARY_VERSION:
                raise ParseError(f"Unsupported binary archive version: {version}")
            self._header_read = True

    def _read_uint8(self) -> int:
        data = self._file.read(1)
        if len(data) < 1:
            raise ParseError("Unexpected end of binary data")
        return struct.unpack("<B", data)[0]

    def _read_uint32(self) -> int:
        data = self._file.read(4)
        if len(data) < 4:
            raise ParseError("Unexpected end of binary data")
        return struct.unpack("<I", data)[0]

    def _read_uint64(self) -> int:
        data = self._file.read(8)
        if len(data) < 8:
            raise ParseError("Unexpected end of binary data")
        return struct.unpack("<Q", data)[0]

    def _read_int32(self) -> int:
        data = self._file.read(4)
        if len(data) < 4:
            raise ParseError("Unexpected end of binary data")
        return struct.unpack("<i", data)[0]

    def _read_int64(self) -> int:
        data = self._file.read(8)
        if len(data) < 8:
            raise ParseError("Unexpected end of binary data")
        return struct.unpack("<q", data)[0]

    def _read_double(self) -> float:
        data = self._file.read(8)
        if len(data) < 8:
            raise ParseError("Unexpected end of binary data")
        return struct.unpack("<d", data)[0]

    def _read_name(self) -> str:
        length = self._read_uint64()
        if length == 0:
            return ""
        data = self._file.read(length)
        if len(data) < length:
            raise ParseError("Unexpected end of binary data while reading name")
        return data.decode("utf-8")

    def _read_string_value(self) -> str:
        length = self._read_uint64()
        if length == 0:
            return ""
        data = self._file.read(length)
        if len(data) < length:
            raise ParseError("Unexpected end of binary data while reading string")
        return data.decode("utf-8")

    def _read_array_value(self, elem_tag: int) -> np.ndarray:
        count = self._read_uint64()
        if count == 0:
            dtype = {ELEM_INT32: np.int32, ELEM_INT64: np.int64, ELEM_FLOAT64: np.float64}.get(elem_tag, np.float64)
            return np.array([], dtype=dtype)

        if elem_tag == ELEM_FLOAT64:
            data = self._file.read(count * 8)
            return np.frombuffer(data, dtype=np.float64).copy()
        elif elem_tag == ELEM_INT32:
            data = self._file.read(count * 4)
            return np.frombuffer(data, dtype=np.int32).copy()
        elif elem_tag == ELEM_INT64:
            data = self._file.read(count * 8)
            return np.frombuffer(data, dtype=np.int64).copy()
        else:
            raise ParseError(f"Unknown array element type tag: {elem_tag}")

    def _read_field(self) -> tuple[str, Any]:
        """Read a single field (name + type tag + value)."""
        name = self._read_name()
        type_tag = self._read_uint8()

        if type_tag == TYPE_INT32:
            return name, self._read_int32()
        elif type_tag == TYPE_INT64:
            return name, self._read_int64()
        elif type_tag == TYPE_FLOAT64:
            return name, self._read_double()
        elif type_tag == TYPE_STRING:
            return name, self._read_string_value()
        elif type_tag == TYPE_ARRAY:
            elem_tag = self._read_uint8()
            return name, self._read_array_value(elem_tag)
        elif type_tag == TYPE_GROUP:
            field_count = self._read_uint64()
            group_data = {}
            for _ in range(field_count):
                field_name, field_value = self._read_field()
                group_data[field_name] = field_value
            return name, group_data
        elif type_tag == TYPE_LIST:
            item_count = self._read_uint64()
            if item_count == 0:
                return name, []
            # Peek at the first type tag to determine list content type
            first_type_tag = self._read_uint8()

            if first_type_tag == TYPE_STRING:
                # List of strings - each item is just a string value
                items = [self._read_string_value()]
                for _ in range(item_count - 1):
                    # Subsequent strings also have TYPE_STRING tag
                    self._read_uint8()  # TYPE_STRING
                    items.append(self._read_string_value())
                return name, items
            elif first_type_tag == TYPE_GROUP:
                # List of compound objects - each item has TYPE_GROUP + field_count + fields
                items = []
                # First item already read TYPE_GROUP
                field_count = self._read_uint64()
                first_item = {}
                for _ in range(field_count):
                    field_name, field_value = self._read_field()
                    first_item[field_name] = field_value
                items.append(first_item)
                # Remaining items
                for _ in range(item_count - 1):
                    self._read_uint8()  # TYPE_GROUP tag
                    fc = self._read_uint64()
                    item = {}
                    for _ in range(fc):
                        fn, fv = self._read_field()
                        item[fn] = fv
                    items.append(item)
                return name, items
            else:
                raise ParseError(f"Unexpected type tag in list: {first_type_tag}")
        else:
            raise ParseError(f"Unknown type tag: {type_tag}")

    def _read_list_item_with_schema(self) -> dict:
        """Read a list item with full field names and type tags."""
        result = {}
        # We need to read fields until we've consumed all of them
        # The tricky part is knowing when to stop - we peek at the stream
        # Actually, in the C++ format, anonymous groups don't have a field count
        # We need to read until we hit the next item or end of list
        # Let me re-examine the format...

        # Looking at the C++ code: anonymous groups within lists don't write
        # their own field count. The first item writes full schema (names + types),
        # subsequent items skip names and types.

        # For Python reading, we need to know the structure. Let's read fields
        # until we can't anymore (this is tricky without a field count).

        # Actually, the C++ writer does write field counts for named groups within
        # list items. Let me trace through more carefully...

        # The serialize template for compound vectors calls:
        # ar.begin_list(name) -> writes name + TYPE_LIST + placeholder count
        # for each elem: ar.begin_group() -> anonymous, increments parent count
        #   serialize each field -> writes name + type + value (first item only)
        # ar.end_group()
        # ar.end_list()

        # So for the first item, each field writes: name + type + value
        # We need to know how many fields. Since this is self-describing,
        # we should be able to detect field boundaries by reading name+type+value
        # repeatedly until we see something that's not a valid field.

        # For simplicity, let's read the first item by peeking: if we can read
        # a name (uint64 length + string), then it's a field. If length looks
        # unreasonable (> 1000) or we hit EOF, we're done.

        while True:
            pos = self._file.tell()
            try:
                length = self._read_uint64()
                if length > 10000:  # Unreasonably long name = probably not a name
                    self._file.seek(pos)
                    break
                if length > 0:
                    name_bytes = self._file.read(length)
                    if len(name_bytes) < length:
                        self._file.seek(pos)
                        break
                    name = name_bytes.decode("utf-8")
                else:
                    name = ""

                # Check if name looks valid (alphanumeric + underscore)
                if name and not all(c.isalnum() or c == '_' for c in name):
                    self._file.seek(pos)
                    break

                type_tag = self._read_uint8()

                if type_tag == TYPE_INT32:
                    result[name] = self._read_int32()
                elif type_tag == TYPE_INT64:
                    result[name] = self._read_int64()
                elif type_tag == TYPE_FLOAT64:
                    result[name] = self._read_double()
                elif type_tag == TYPE_STRING:
                    result[name] = self._read_string_value()
                elif type_tag == TYPE_ARRAY:
                    elem_tag = self._read_uint8()
                    result[name] = self._read_array_value(elem_tag)
                elif type_tag == TYPE_GROUP:
                    field_count = self._read_uint64()
                    group_data = {}
                    for _ in range(field_count):
                        field_name, field_value = self._read_field()
                        group_data[field_name] = field_value
                    result[name] = group_data
                else:
                    # Unknown type tag - probably not a field, rewind
                    self._file.seek(pos)
                    break
            except:
                self._file.seek(pos)
                break

        return result

    def _extract_schema(self, item: dict) -> list[tuple[str, int, Any]]:
        """Extract schema from first list item for reading subsequent items."""
        schema = []
        for name, value in item.items():
            if isinstance(value, (int, np.integer)):
                if isinstance(value, np.int32) or (isinstance(value, int) and -2**31 <= value < 2**31):
                    schema.append((name, TYPE_INT32, None))
                else:
                    schema.append((name, TYPE_INT64, None))
            elif isinstance(value, (float, np.floating)):
                schema.append((name, TYPE_FLOAT64, None))
            elif isinstance(value, str):
                schema.append((name, TYPE_STRING, None))
            elif isinstance(value, np.ndarray):
                if value.dtype == np.float64:
                    schema.append((name, TYPE_ARRAY, ELEM_FLOAT64))
                elif value.dtype == np.int32:
                    schema.append((name, TYPE_ARRAY, ELEM_INT32))
                elif value.dtype == np.int64:
                    schema.append((name, TYPE_ARRAY, ELEM_INT64))
                else:
                    schema.append((name, TYPE_ARRAY, ELEM_FLOAT64))
            elif isinstance(value, dict):
                # Nested group - we need to recurse
                schema.append((name, TYPE_GROUP, self._extract_schema(value)))
        return schema

    def _read_list_item_without_schema(self, schema: list) -> dict:
        """Read a list item using known schema (no names/types in stream)."""
        result = {}
        for name, type_tag, extra in schema:
            if type_tag == TYPE_INT32:
                result[name] = self._read_int32()
            elif type_tag == TYPE_INT64:
                result[name] = self._read_int64()
            elif type_tag == TYPE_FLOAT64:
                result[name] = self._read_double()
            elif type_tag == TYPE_STRING:
                result[name] = self._read_string_value()
            elif type_tag == TYPE_ARRAY:
                result[name] = self._read_array_value(extra)
            elif type_tag == TYPE_GROUP:
                result[name] = self._read_list_item_without_schema(extra)
        return result

    def read_all(self) -> dict:
        """Read entire binary archive into nested dict.

        This handles files that contain multiple concatenated binary archives
        (e.g., products files where each product is serialized with a separate header).
        """
        self._ensure_header()
        result = {}

        while True:
            # Check if we're at EOF
            pos = self._file.tell()
            peek = self._file.read(1)
            if not peek:
                break
            self._file.seek(pos)

            # Check if we hit another header (concatenated archives)
            if len(peek) >= 1:
                # Peek at potential magic number
                potential_magic = self._file.read(4)
                if len(potential_magic) == 4:
                    magic_val = struct.unpack("<I", potential_magic)[0]
                    if magic_val == BINARY_MAGIC:
                        # Skip the header (magic already read, now read version)
                        self._file.read(1)  # version byte
                        continue
                # Not a magic header, rewind
                self._file.seek(pos)

            name, value = self._read_field()
            result[name] = value

        return result


class BinaryWriter:
    """Writer for mist self-describing binary archive format."""

    def __init__(self, dest: Union[str, Path, BinaryIO]):
        if isinstance(dest, (str, Path)):
            self._file = open(dest, "wb")
            self._owns_file = True
        else:
            self._file = dest
            self._owns_file = False
        self._header_written = False
        self._group_positions: list[int] = []
        self._field_counts: list[int] = []
        self._in_anonymous_group = 0
        self._list_item_indices: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._owns_file:
            self._file.close()

    def _ensure_header(self):
        if not self._header_written:
            self._file.write(struct.pack("<I", BINARY_MAGIC))
            self._file.write(struct.pack("<B", BINARY_VERSION))
            self._header_written = True
        # Increment field count for parent group (only if not in anonymous group)
        if self._field_counts and self._in_anonymous_group == 0:
            self._field_counts[-1] += 1

    def _write_name(self, name: str):
        if self._in_anonymous_group > 0:
            return
        encoded = name.encode("utf-8")
        self._file.write(struct.pack("<Q", len(encoded)))
        self._file.write(encoded)

    def _write_type_tag(self, tag: int):
        if self._in_anonymous_group > 0:
            return
        self._file.write(struct.pack("<B", tag))

    def _write_uint64(self, value: int):
        self._file.write(struct.pack("<Q", value))

    def _write_field(self, name: str, value: Any):
        """Write a single field with name, type tag, and value."""
        self._ensure_header()

        if isinstance(value, dict):
            self._write_name(name)
            self._write_type_tag(TYPE_GROUP)
            # Save position for field count
            pos = self._file.tell()
            self._write_uint64(0)  # Placeholder
            count = 0
            for k, v in value.items():
                # Temporarily disable anonymous mode to write nested fields
                saved_anon = self._in_anonymous_group
                self._in_anonymous_group = 0
                self._write_field(k, v)
                self._in_anonymous_group = saved_anon
                count += 1
            # Backfill count
            end_pos = self._file.tell()
            self._file.seek(pos)
            self._write_uint64(count)
            self._file.seek(end_pos)

        elif isinstance(value, list):
            self._write_name(name)
            self._write_type_tag(TYPE_LIST)
            pos = self._file.tell()
            self._write_uint64(0)  # Placeholder for count

            if len(value) == 0:
                # Empty list - just backfill count = 0
                pass
            elif isinstance(value[0], str):
                # List of strings - each item is TYPE_STRING + length + data
                for item in value:
                    self._file.write(struct.pack("<B", TYPE_STRING))
                    encoded = item.encode("utf-8")
                    self._write_uint64(len(encoded))
                    self._file.write(encoded)
            elif isinstance(value[0], dict):
                # List of compound objects
                self._list_item_indices.append(0)

                for i, item in enumerate(value):
                    self._list_item_indices[-1] = i + 1
                    is_first = (i == 0)
                    if not is_first:
                        self._in_anonymous_group += 1

                    # Write item fields
                    for k, v in item.items():
                        self._write_field(k, v)

                    if not is_first:
                        self._in_anonymous_group -= 1

                self._list_item_indices.pop()
            else:
                raise TypeError(f"Unsupported list item type: {type(value[0])}")

            # Backfill count
            end_pos = self._file.tell()
            self._file.seek(pos)
            self._write_uint64(len(value))
            self._file.seek(end_pos)

        elif isinstance(value, np.ndarray):
            self._write_name(name)
            self._write_type_tag(TYPE_ARRAY)
            if value.dtype == np.float64 or np.issubdtype(value.dtype, np.floating):
                self._write_type_tag(ELEM_FLOAT64)
                self._write_uint64(len(value))
                self._file.write(value.astype(np.float64).tobytes())
            elif value.dtype == np.int32:
                self._write_type_tag(ELEM_INT32)
                self._write_uint64(len(value))
                self._file.write(value.tobytes())
            elif value.dtype == np.int64 or np.issubdtype(value.dtype, np.integer):
                # Check if values fit in int32
                if value.min() >= -2**31 and value.max() < 2**31:
                    self._write_type_tag(ELEM_INT32)
                    self._write_uint64(len(value))
                    self._file.write(value.astype(np.int32).tobytes())
                else:
                    self._write_type_tag(ELEM_INT64)
                    self._write_uint64(len(value))
                    self._file.write(value.astype(np.int64).tobytes())
            else:
                # Default to float64
                self._write_type_tag(ELEM_FLOAT64)
                self._write_uint64(len(value))
                self._file.write(value.astype(np.float64).tobytes())

        elif isinstance(value, str):
            self._write_name(name)
            self._write_type_tag(TYPE_STRING)
            encoded = value.encode("utf-8")
            self._write_uint64(len(encoded))
            self._file.write(encoded)

        elif isinstance(value, float) or isinstance(value, np.floating):
            self._write_name(name)
            self._write_type_tag(TYPE_FLOAT64)
            self._file.write(struct.pack("<d", float(value)))

        elif isinstance(value, (int, np.integer)):
            self._write_name(name)
            if -2**31 <= value < 2**31:
                self._write_type_tag(TYPE_INT32)
                self._file.write(struct.pack("<i", int(value)))
            else:
                self._write_type_tag(TYPE_INT64)
                self._file.write(struct.pack("<q", int(value)))
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def write_all(self, data: dict):
        """Write entire nested dict to binary archive."""
        for name, value in data.items():
            self._write_field(name, value)


# Convenience functions


def load(source: Union[str, Path, TextIO, BinaryIO]) -> dict:
    """Load an archive file into a nested dictionary.

    Format is auto-detected by:
    - File extension (.bin for binary) for path strings
    - File mode ('b' in mode) for file objects

    Args:
        source: File path (str or Path) or file-like object

    Returns:
        Nested dictionary with archive contents

    Example:
        config = ma.load("config.cfg")
        checkpoint = ma.load("chkpt.0000.dat")
        binary_data = ma.load("data.bin")

        # Also works with file objects
        with open("data.bin", "rb") as f:
            data = ma.load(f)
    """
    # For path strings, detect by extension
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".bin":
            with BinaryReader(source) as reader:
                return reader.read_all()
        with AsciiReader(source) as reader:
            return reader.read_all()

    # For file objects, detect by mode
    if hasattr(source, 'mode') and 'b' in source.mode:
        with BinaryReader(source) as reader:
            return reader.read_all()

    with AsciiReader(source) as reader:
        return reader.read_all()


def load_ascii(source: Union[str, Path, TextIO]) -> dict:
    """Load an ASCII archive file into a nested dictionary.

    Args:
        source: File path (str or Path) or file-like object

    Returns:
        Nested dictionary with archive contents

    Example:
        config = ma.load_ascii("config.cfg")
    """
    with AsciiReader(source) as reader:
        return reader.read_all()


def dump(data: dict, dest: Union[str, Path, TextIO], indent_size: int = 4):
    """Write a nested dictionary to an archive file.

    Format is auto-detected by file extension:
    - .bin: Binary format
    - .cfg, .dat, or other: ASCII format

    Args:
        data: Nested dictionary to serialize
        dest: File path (str or Path) or file-like object
        indent_size: Number of spaces per indentation level for ASCII (default: 4)

    Example:
        ma.dump(config, "config.cfg")
        ma.dump(checkpoint, "chkpt.0001.dat")
        ma.dump(data, "output.bin")  # Binary format
    """
    # Check for binary extension
    if isinstance(dest, (str, Path)):
        path = Path(dest)
        if path.suffix == ".bin":
            with BinaryWriter(dest) as writer:
                writer.write_all(data)
            return

    with AsciiWriter(dest, indent_size) as writer:
        writer.write_all(data)


def dump_binary(data: dict, dest: Union[str, Path, BinaryIO]):
    """Write a nested dictionary to a binary archive file.

    Args:
        data: Nested dictionary to serialize
        dest: File path (str or Path) or file-like object

    Example:
        ma.dump_binary(data, "output.bin")
    """
    with BinaryWriter(dest) as writer:
        writer.write_all(data)


def load_binary(source: Union[str, Path, BinaryIO]) -> dict:
    """Load a binary archive file into a nested dictionary.

    Args:
        source: File path (str or Path) or file-like object

    Returns:
        Nested dictionary with archive contents

    Example:
        data = ma.load_binary("data.bin")
    """
    with BinaryReader(source) as reader:
        return reader.read_all()


def dump_ascii(data: dict, dest: Union[str, Path, TextIO], indent_size: int = 4):
    """Write a nested dictionary to an ASCII archive file.

    Args:
        data: Nested dictionary to serialize
        dest: File path (str or Path) or file-like object
        indent_size: Number of spaces per indentation level (default: 4)

    Example:
        ma.dump_ascii(config, "config.cfg")
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
