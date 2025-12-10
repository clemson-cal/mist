# Serialization Framework Design

## Overview

This document specifies a key-based serialization/deserialization framework supporting multiple backends (ASCII, binary, HDF5). The design prioritizes:

1. **Key-based field access** - Fields identified by name, not position
2. **Forward compatibility** - Missing fields ignored on read (use defaults)
3. **Backward compatibility** - Extra fields in archive ignored
4. **Format agnosticism** - Same serialize/deserialize code works with any backend

## Archive Backends

### Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| ASCII  | `.dat`    | Human-readable, debugging, small data |
| Binary | `.bin`    | Fast I/O, medium data |
| HDF5   | `.h5`     | Large arrays, parallel I/O, self-describing |

### Backend Selection

```cpp
enum class archive_format {
    ascii,
    binary,
    hdf5,
};

// Factory function
auto make_writer(std::ostream& os, archive_format fmt) -> std::unique_ptr<archive_writer>;
auto make_reader(std::istream& is, archive_format fmt) -> std::unique_ptr<archive_reader>;

// HDF5 uses file path instead of stream
auto make_hdf5_writer(const std::string& path) -> std::unique_ptr<archive_writer>;
auto make_hdf5_reader(const std::string& path) -> std::unique_ptr<archive_reader>;
```

## Archive Interface

### Writer Concept

```cpp
template<typename A>
concept ArchiveWriter = requires(A& ar, const char* name) {
    // Scalars
    { ar.write_scalar(name, int{}) };
    { ar.write_scalar(name, double{}) };
    { ar.write_string(name, std::string{}) };

    // Arrays (small, inline)
    { ar.write_array(name, vec_t<double, 3>{}) };
    { ar.write_array(name, std::vector<int>{}) };

    // Bulk data (large arrays)
    { ar.write_data(name, (double*)nullptr, std::size_t{}) };

    // Structure
    { ar.begin_group(name) };
    { ar.begin_group() };  // anonymous group (for lists)
    { ar.end_group() };
    { ar.begin_list(name) };
    { ar.end_list() };
};
```

### Reader Concept

```cpp
template<typename A>
concept ArchiveReader = requires(A& ar, const char* name, int& i, double& d) {
    // Scalars - return true if field exists
    { ar.read_scalar(name, i) } -> std::same_as<bool>;
    { ar.read_scalar(name, d) } -> std::same_as<bool>;
    { ar.read_string(name, std::string{}) } -> std::same_as<bool>;

    // Arrays
    { ar.read_array(name, vec_t<double, 3>{}) } -> std::same_as<bool>;
    { ar.read_array(name, std::vector<int>{}) } -> std::same_as<bool>;

    // Bulk data
    { ar.read_data(name, (double*)nullptr, std::size_t{}) } -> std::same_as<bool>;

    // Structure
    { ar.begin_group(name) } -> std::same_as<bool>;
    { ar.begin_group() } -> std::same_as<bool>;
    { ar.end_group() };
    { ar.begin_list(name) } -> std::same_as<bool>;
    { ar.end_list() };

    // Query
    { ar.has_field(name) } -> std::same_as<bool>;
    { ar.count_items(name) } -> std::same_as<std::size_t>;
    { ar.field_names() } -> std::convertible_to<std::vector<std::string>>;
};
```

## Key-Based Deserialization

### Current Problem

The current implementation reads fields in strict order:

```cpp
// Current: order-dependent
template<ArchiveReader A, typename T>
void deserialize(A& ar, const char* name, T& value) {
    ar.begin_group(name);
    std::apply([&ar](auto&&... fields) {
        (deserialize(ar, fields.name, fields.value), ...);  // Must match write order!
    }, value.fields());
    ar.end_group();
}
```

### Proposed Solution

Read fields by key lookup, ignore missing fields:

```cpp
template<ArchiveReader A, typename T>
    requires HasFields<T>
void deserialize(A& ar, const char* name, T& value) {
    if (!ar.begin_group(name)) {
        return;  // Group missing - keep defaults
    }

    std::apply([&ar](auto&&... fields) {
        (deserialize_field(ar, fields.name, fields.value), ...);
    }, value.fields());

    ar.end_group();
}

template<ArchiveReader A, typename T>
void deserialize_field(A& ar, const char* name, T& value) {
    if (ar.has_field(name)) {
        deserialize(ar, name, value);
    }
    // else: field missing, keep default value
}
```

### Reader Implementation Changes

Each backend must support:

1. **`has_field(name)`** - Check if field exists in current group
2. **`field_names()`** - List all field names in current group (for debugging)
3. **Non-positional reads** - `read_scalar(name, value)` finds field by name

#### ASCII Backend

The ASCII reader needs to:
1. On `begin_group()`: scan and index all fields in the group
2. Store field positions in a map
3. On `read_scalar(name, ...)`: seek to field position, read value

```cpp
class ascii_reader {
    struct group_index_t {
        std::map<std::string, std::streampos> fields;
        std::map<std::string, std::streampos> groups;
        std::streampos end_pos;
    };

    std::vector<group_index_t> group_stack_;

    void begin_group(const char* name) {
        // 1. Find and enter the named group
        // 2. Scan contents to build index
        // 3. Push index onto stack
    }

    bool has_field(const char* name) {
        return group_stack_.back().fields.contains(name);
    }

    template<typename T>
    bool read_scalar(const char* name, T& value) {
        auto& idx = group_stack_.back();
        auto it = idx.fields.find(name);
        if (it == idx.fields.end()) {
            return false;
        }
        is_.seekg(it->second);
        // ... read value
        return true;
    }
};
```

#### Binary Backend

Binary format stores a table of contents (TOC) at the start of each group:

```
[group]
  u32: num_fields
  [field_entry] * num_fields:
    u16: name_length
    char[name_length]: field_name
    u64: offset_from_group_start
    u32: data_size
  [field_data] * num_fields:
    ... actual data at specified offsets
```

#### HDF5 Backend

HDF5 natively supports key-based access - groups and datasets are accessed by name.

```cpp
class hdf5_reader {
    hid_t file_;
    std::vector<hid_t> group_stack_;

    bool has_field(const char* name) {
        return H5Lexists(group_stack_.back(), name, H5P_DEFAULT) > 0;
    }

    template<typename T>
    bool read_scalar(const char* name, T& value) {
        if (!has_field(name)) return false;
        hid_t dset = H5Dopen(group_stack_.back(), name, H5P_DEFAULT);
        // ... read value
        H5Dclose(dset);
        return true;
    }
};
```

## File Format Specifications

### ASCII Format

Human-readable, indented structure:

```
state {
    time = 1.5
    iteration = 100

    conserved {
        start = [0]
        shape = [200]
        data = [0.1, 0.2, 0.3, ...]
    }

    patches {
        {
            rank = 0
            conserved {
                start = [0]
                shape = [50]
                data = [...]
            }
        }
        {
            rank = 1
            conserved {
                start = [50]
                shape = [50]
                data = [...]
            }
        }
    }
}
```

Key features:
- Fields are `name = value`
- Groups are `name { ... }`
- Anonymous groups (in lists) are `{ ... }`
- Comments start with `#`
- Order of fields within a group is not significant

### Binary Format

```
[file header]
  magic: "MIST" (4 bytes)
  version: u16 (format version)
  flags: u16 (endianness, compression)

[group: root]
  [TOC]
    num_entries: u32
    entries[]:
      name_len: u16
      name: char[name_len]
      type: u8 (scalar, array, group, data)
      offset: u64
      size: u64
  [data section]
    ... field data at specified offsets
```

### HDF5 Format

Standard HDF5 structure:
- Groups map to HDF5 groups
- Scalars stored as scalar datasets or attributes
- Arrays stored as datasets with appropriate shape
- Bulk data stored as datasets with chunking/compression

```
/state (group)
  /state/time (scalar dataset, float64)
  /state/iteration (scalar dataset, int32)
  /state/patches (group)
    /state/patches/0 (group)
      /state/patches/0/rank (attribute or dataset)
      /state/patches/0/conserved (dataset, float64[50])
    /state/patches/1 (group)
      ...
```

## Usage Examples

### Writing

```cpp
void save_state(const state_t& state, const std::string& path, archive_format fmt) {
    if (fmt == archive_format::hdf5) {
        auto ar = make_hdf5_writer(path);
        serialize(*ar, "state", state);
    } else {
        std::ofstream ofs(path);
        auto ar = make_writer(ofs, fmt);
        serialize(*ar, "state", state);
    }
}
```

### Reading with Missing Fields

```cpp
struct config_t {
    int rk_order = 2;           // default
    double cfl = 0.4;           // default
    std::string method = "hll"; // default (new field)

    auto fields() { return std::make_tuple(
        field("rk_order", rk_order),
        field("cfl", cfl),
        field("method", method)
    ); }
};

// Reading old file without "method" field:
// - rk_order and cfl are read from file
// - method keeps its default value "hll"
```

### Custom Serialization

For types that need special handling (like `patch_state_t`):

```cpp
// Serialize only the conserved array, not internal state
template<ArchiveWriter A>
void serialize(A& ar, const char* name, const patch_state_t& patch) {
    ar.begin_group(name);
    serialize(ar, "rank", patch.rank());
    serialize(ar, "npartitions", patch.npartitions());
    serialize(ar, "conserved", patch.conserved());
    ar.end_group();
}

// Deserialize and reconstruct
template<ArchiveReader A>
void deserialize(A& ar, const char* name, patch_state_t& patch) {
    if (!ar.begin_group(name)) return;

    int rank = 0, npartitions = 1;
    cached_t<double, 1> conserved;

    deserialize_field(ar, "rank", rank);
    deserialize_field(ar, "npartitions", npartitions);
    deserialize_field(ar, "conserved", conserved);

    patch = patch_state_t(rank, npartitions, std::move(conserved));
    ar.end_group();
}
```

## Migration Path

### Phase 1: Update Reader Interface
1. Add `has_field()` and `field_names()` to readers
2. Change `read_*` methods to return `bool`
3. Update `deserialize()` to use `deserialize_field()` pattern

### Phase 2: Update ASCII Reader
1. Implement group indexing on `begin_group()`
2. Support out-of-order field reads

### Phase 3: Add Binary Backend
1. Implement binary writer with TOC
2. Implement binary reader with TOC-based lookup

### Phase 4: Add HDF5 Backend
1. Implement HDF5 writer
2. Implement HDF5 reader
3. Add compression options

## Error Handling

### Strict Mode vs Lenient Mode

```cpp
enum class read_mode {
    strict,   // Missing required fields throw
    lenient,  // Missing fields use defaults
};

template<ArchiveReader A, typename T>
void deserialize(A& ar, const char* name, T& value, read_mode mode = read_mode::lenient);
```

### Required Fields

Mark fields as required using a wrapper:

```cpp
template<typename T>
struct required_t {
    T& value;
};

template<typename T>
required_t<T> required(T& value) { return {value}; }

// Usage:
auto fields() {
    return std::make_tuple(
        field("time", required(time)),  // Must exist
        field("cfl", cfl)               // Optional, has default
    );
}
```
