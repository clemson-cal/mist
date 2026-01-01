# Archive

A C++20 concept-based serialization library for HPC applications. Designed for checkpointing distributed simulations with shared metadata and distributed patch data.

## Features

- **Key-based access**: Forward/backward compatible archives
- **Binary format**: Self-describing, efficient for large data
- **ASCII format**: Human-readable, debuggable
- **ADL-based**: Types opt-in via free `fields()` function
- **Truthful types**: Separate runtime state from serialized state
- **Parallel I/O**: Checkpoint/restart with MPI-style scatter/gather

## Files

```
archive.hpp      # Umbrella header (include this)
├── checkpoint.hpp   # Layer 2: Checkpointable protocol for parallel I/O
├── protocol.hpp     # Layer 0: write/read functions, HasFields concept
├── sink.hpp         # ascii_sink, binary_sink classes
├── source.hpp       # ascii_source, binary_source classes
└── format.hpp       # Binary format constants
```

## Quick Start

```cpp
#include "archive.hpp"
#include <fstream>

struct config {
    int resolution = 100;
    double cfl = 0.4;
    std::string method = "hll";
};

// ADL free function (required for serialization)
auto fields(config& c) {
    return std::make_tuple(
        archive::field("resolution", c.resolution),
        archive::field("cfl", c.cfl),
        archive::field("method", c.method)
    );
}
auto fields(const config& c) {
    return std::make_tuple(
        archive::field("resolution", c.resolution),
        archive::field("cfl", c.cfl),
        archive::field("method", c.method)
    );
}

// Write
std::ofstream out("config.bin", std::ios::binary);
archive::binary_sink sink(out);
archive::write(sink, "config", cfg);

// Read
std::ifstream in("config.bin", std::ios::binary);
archive::binary_source source(in);
config cfg;
archive::read(source, "config", cfg);
```

## Making Types Serializable

### Simple Structs

Define a free `fields()` function returning a tuple of `archive::field()` pairs:

```cpp
struct vec3 {
    double x, y, z;
};

auto fields(vec3& v) {
    return std::make_tuple(
        archive::field("x", v.x),
        archive::field("y", v.y),
        archive::field("z", v.z)
    );
}
auto fields(const vec3& v) {
    return std::make_tuple(
        archive::field("x", v.x),
        archive::field("y", v.y),
        archive::field("z", v.z)
    );
}
```

### Enums

Define `to_string` and `from_string` via ADL:

```cpp
enum class boundary { periodic, outflow, reflecting };

const char* to_string(boundary b) {
    switch (b) {
        case boundary::periodic: return "periodic";
        case boundary::outflow: return "outflow";
        case boundary::reflecting: return "reflecting";
    }
}

boundary from_string(std::type_identity<boundary>, const std::string& s) {
    if (s == "periodic") return boundary::periodic;
    if (s == "outflow") return boundary::outflow;
    if (s == "reflecting") return boundary::reflecting;
    throw std::runtime_error("unknown boundary: " + s);
}
```

### Truthful Types (Runtime State)

For types with transient runtime state that shouldn't be serialized:

```cpp
struct simulation_state {
    double time;
    int iteration;
    std::vector<double> data;

    // Runtime-only (not serialized)
    bool dirty_flag;
    std::optional<cached_result> cache;
};

struct simulation_truth {
    double time;
    int iteration;
    std::vector<double> data;
};

auto fields(simulation_truth& t) { /* ... */ }
auto fields(const simulation_truth& t) { /* ... */ }

// ADL functions for Truthful concept
auto to_truth(const simulation_state& s) -> simulation_truth {
    return {s.time, s.iteration, s.data};
}

void from_truth(simulation_state& s, simulation_truth t) {
    s.time = t.time;
    s.iteration = t.iteration;
    s.data = std::move(t.data);
    s.dirty_flag = false;
    s.cache = std::nullopt;
}
```

## Supported Types

Built-in `write`/`read` overloads for:

- Arithmetic types (`int`, `double`, etc.)
- `std::string`
- `std::vector<T>` (arithmetic T uses efficient bulk I/O)
- `std::map<std::string, T>`
- `std::pair<T1, T2>`
- `std::optional<T>`
- `std::variant<Ts...>`
- Any type with ADL `fields()` function
- Any type satisfying `Truthful` concept

## Integration with HPC Framework

### Checkpoint Directory Structure

```
checkpoint_0001/
├── header.bin          # Shared metadata (all ranks read)
└── patches/
    ├── block_0_0_0.bin  # Patch data (one rank reads each)
    ├── block_0_0_1.bin
    └── ...
```

### Making a State Checkpointable

A state type must satisfy `Checkpointable = Emittable && Scatterable && Gatherable`:

```cpp
struct app_state {
    using patch_key_type = block_index;  // Required typedef

    // Shared metadata
    double time;
    int iteration;
    mesh_config mesh;

    // Distributed data
    std::unordered_map<block_index, block_data> blocks;
};

// --- Emittable: shared data ---

void emit(archive::Sink auto& sink, const app_state& s) {
    sink.begin_group();
    archive::write(sink, "time", s.time);
    archive::write(sink, "iteration", s.iteration);
    archive::write(sink, "mesh", s.mesh);
    sink.end_group();
}

void load(archive::Source auto& source, app_state& s) {
    source.begin_group();
    archive::read(source, "time", s.time);
    archive::read(source, "iteration", s.iteration);
    archive::read(source, "mesh", s.mesh);
    source.end_group();
}

// --- Scatterable: enumerate and access patches ---

auto patch_keys(const app_state& s) {
    return s.blocks | std::views::keys;
}

const block_data& patch_data(const app_state& s, block_index key) {
    return s.blocks.at(key);
}

// --- Gatherable: ownership and insertion ---

bool patch_affinity(const app_state& s, block_index key, int rank, int num_ranks) {
    // Partition patches among ranks (e.g., by hash)
    return std::hash<block_index>{}(key) % num_ranks == rank;
}

void emplace_patch(app_state& s, block_index key, archive::Source auto& source) {
    block_data data;
    archive::read(source, data);
    s.blocks[key] = std::move(data);
}
```

### Patch Key Requirements

The `patch_key_type` must satisfy `PatchKey`:

```cpp
struct block_index {
    int level, i, j, k;
    bool operator==(const block_index&) const = default;
};

// For filename generation
std::string to_string(block_index idx) {
    return std::format("L{}_{}_{}_{}", idx.level, idx.i, idx.j, idx.k);
}

block_index from_string(std::type_identity<block_index>, std::string_view s) {
    // Parse "L0_1_2_3" format
    block_index idx;
    std::sscanf(s.data(), "L%d_%d_%d_%d", &idx.level, &idx.i, &idx.j, &idx.k);
    return idx;
}

// For std::unordered_map
template<> struct std::hash<block_index> {
    size_t operator()(block_index idx) const {
        return std::hash<int>{}(idx.level) ^ (std::hash<int>{}(idx.i) << 1)
             ^ (std::hash<int>{}(idx.j) << 2) ^ (std::hash<int>{}(idx.k) << 3);
    }
};

// fields() for serialization
auto fields(block_index& b) {
    return std::make_tuple(
        archive::field("level", b.level),
        archive::field("i", b.i),
        archive::field("j", b.j),
        archive::field("k", b.k)
    );
}
auto fields(const block_index& b) { /* same */ }
```

### Distributed I/O API

```cpp
// Write distributed data (each rank writes its patches)
archive::distributed_write("checkpoint_0001", state, archive::Binary{});

// Read distributed data (patches distributed by affinity)
archive::distributed_read("checkpoint_0001", state, rank, num_ranks, archive::Binary{});
```

## Binary Format

Self-describing format with magic header `MIST` (0x4D495354):

- Header: `u32 magic` + `u8 version`
- Fields: `u64 name_len` + `name` + `u8 type_tag` + data
- Groups: type tag + `u64 field_count` + fields
- Arrays: type tag + `u8 elem_type` + `u64 count` + elements

## ASCII Format

Human-readable key-value format:

```
config {
    resolution = 100
    cfl = 0.4
    method = "hll"

    mesh {
        dimensions = [100, 100, 100]
        origin = [0.0, 0.0, 0.0]
    }
}
```

## Design Decisions

1. **ADL free functions only** - No member function `fields()`, only free functions
2. **Key-based archives** - Fields read by name, not position
3. **`read()` returns bool** - Missing fields return false, enabling defaults
4. **Truthful separation** - Runtime state excluded via `to_truth`/`from_truth`
5. **Patches are any Writable** - Not limited to arrays
6. **No index file** - Patch keys recovered from directory listing
7. **Affinity predicate** - Caller controls patch distribution on read
