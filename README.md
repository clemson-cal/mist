# Mist

A lightweight C++20 header-only library for deploy-anywhere array transformations and physics simulations.

Mist is an evolution of [Vapor](https://github.com/clemson-cal/vapor), a library with similar goals for HPC physics applications.

## Features

- **Core library** (`mist/core.hpp`): Multi-dimensional arrays, index spaces, transforms, and parallel execution
- **Driver library** (`mist/driver.hpp`): Time-stepping and scheduled output management for physics simulations
- **Header-only**: No compilation required, just include and go
- **CUDA compatible**: All functions work on both CPU and GPU (CUDA 12+)
- **Zero dependencies**: Pure C++20 standard library

## Quick Start

```bash
# Copy headers to your project
cp -r include/mist /path/to/your/project/

# Or clone and use directly
git clone https://github.com/yourusername/mist.git
cd mist/examples/advection-1d
make
./advection-1d
```

## CUDA Compatibility

Mist is fully compatible with CUDA 12+ and can be used in both host and device code. All functions are annotated with `__host__ __device__` when compiled with nvcc.

## Data structures
All data structures use overloaded free functions rather than member functions. Data members are public but start with an underscore.

- `vec_t<T, S>` is a statically sized array type with element type `T` and size `S`
  * aliases `dvec_t`, `ivec_t` and, `uvec_t` for types `double`, `int`, and `unsigned int`
  * constructor e.g. `vec(0.4, 1.0, 2.0) -> vec_t<double, 3>` detects `T` using `std::common_type`
  * constructors
    + `dvec`
    + `ivec`
    + `uvec`
    + `range<S> -> uvec_t<S>`
  * operators
    + `vec_t<T, S> + vec_t<U, S>`
    + `vec_t<T, S> - vec_t<U, S>`
    + `vec_t<T, S> * U` and `U * vec_t<T, S>`
    + `vec_t<T, S> / U`
    + `==`
  * free functions
    + `dot(vec<T, S>, vec<U, S>)` - dot product
    + `map(vec_t<T, S>, [] (T) -> U { ... })` - element-wise transformation
    + `sum(vec_t<T, S>)` - sum of all elements, returns `T`
    + `product(vec_t<T, S>)` - product of all elements, returns `T`
    + `any(vec_t<bool, S>)` - true if any element is true, returns `bool`
    + `all(vec_t<bool, S>)` - true if all elements are true, returns `bool`
- `index_space_t<S>` is `start: ivec_t<S>` and a `shape: uvec_t<S>`
  * constructors
    + `index_space(start, shape)` - creates an index_space_t with given start and shape
    + Example: `auto space = index_space(ivec(0, 0), uvec(10, 20));`
  * operators
    + `==`
  * free functions
    + `start(space)` - returns `_start`
    + `shape(space)` - returns `_shape`
    + `size(space)` - returns total number of elements
    + `contains(space, index)` - returns `bool`, checks if index is within bounds
  * iteration
    + `begin(space)` and `end(space)` - iterate over all indices in the space
    + Example: `for (auto index : space) { ... }` iterates through all valid indices
    + `for_each(space, func)` - execute `func(index)` for every index in the space
    + `for_each(space, func, exec_mode)` where `exec_mode` is one of `exec::cpu`, `exec::omp`, `exec::gpu`
  * reductions
    + `map_reduce(space, init, map, reduce_op)` - map-reduce over all indices
    + `map_reduce(space, init, map, reduce_op, exec_mode)` - with specified execution policy
    + `map` signature: `(ivec_t<S> index) -> T` - transforms index to value
    + `reduce_op` signature: `(T, T) -> T` - binary associative reduction operator
    + Example: `auto sum = map_reduce(space, 0, [&buf](auto idx) { return ndread(buf, space, idx); }, std::plus<>{});`
    + Example: `auto max = map_reduce(space, -INF, [&buf](auto idx) { return ndread(buf, space, idx); }, [](auto a, auto b) { return std::max(a, b); });`

## Multi-dimensional indexing functions
All indices are **absolute** (relative to the origin `[0, 0, ...]`). For example, with `_start = [5, 10]` and `_shape = [10, 20]`, valid indices range from `[5, 10]` to `[14, 29]` (i.e., `_start` to `_start + _shape - 1`).

- `ndoffset(space, index)` - Compute flat buffer offset from multi-dimensional index
  * `space: index_space_t<S>` - The index space defining the array dimensions
  * `index: ivec_t<S>` - Absolute multi-dimensional index
  * Returns `std::size_t` - Flat offset into buffer (row-major ordering)
  * Example: `ndoffset(space, ivec(7, 13))` with `start = [5, 10]` and `shape = [10, 20]` returns `(7-5) * 20 + (13-10) = 43`

- `ndindex(space, offset)` - Convert flat buffer offset to multi-dimensional index
  * `space: index_space_t<S>` - The index space defining the array dimensions
  * `offset: std::size_t` - Flat offset into buffer
  * Returns `ivec_t<S>` - Absolute multi-dimensional index
  * Inverse of `ndoffset`: `ndindex(space, ndoffset(space, index)) == index`

- `ndread(data, space, index)` - Read element from buffer using multi-dimensional index
  * `data: const T*` - Pointer to flat buffer
  * `space: index_space_t<S>` - Index space defining dimensions
  * `index: ivec_t<S>` - Multi-dimensional index
  * Returns `T` - Value at that index
  * Example: `ndread(buffer, space, ivec(1, 2))` reads element at row 1, column 2

- `ndwrite(data, space, index, value)` - Write element to buffer using multi-dimensional index
  * `data: T*` - Pointer to flat buffer
  * `space: index_space_t<S>` - Index space defining dimensions
  * `index: ivec_t<S>` - Multi-dimensional index
  * `value: T` - Value to write
  * Example: `ndwrite(buffer, space, ivec(1, 2), 42.0)` writes to row 1, column 2

- `ndread_soa<T, N>(data, space, index)` - Read a `vec_t<T, N>` from SoA buffer
  * Template parameters: `T` (element type), `N` (vector size)
  * `data: const T*` - Pointer to flat buffer with all components stored contiguously
  * `space: index_space_t<S>` - Index space defining spatial dimensions
  * `index: ivec_t<S>` - Multi-dimensional spatial index
  * Returns `vec_t<T, N>` - Vector with components gathered from memory
  * Layout: Component `i` of vector at `index` is at offset `i * size(space) + ndoffset(space, index)`
  * Component-major ordering: For a 2D grid `[10, 20]` storing 3-vectors (200 total positions):
    - All x-components: `[x₀, x₁, x₂, ..., x₁₉₉]`
    - All y-components: `[y₀, y₁, y₂, ..., y₁₉₉]`
    - All z-components: `[z₀, z₁, z₂, ..., z₁₉₉]`
  * Usage: `auto v = ndread_soa<double, 3>(buffer, space, ivec(1, 2));`

- `ndwrite_soa<T, N>(data, space, index, value)` - Write a `vec_t<T, N>` to SoA buffer
  * Template parameters: `T` (element type), `N` (vector size)
  * `data: T*` - Pointer to flat buffer
  * `space: index_space_t<S>` - Index space defining spatial dimensions
  * `index: ivec_t<S>` - Multi-dimensional spatial index
  * `value: vec_t<T, N>` - Vector to write
  * Scatters vector components into memory with same layout as `ndread_soa`
  * Usage: `ndwrite_soa<double, 3>(buffer, space, ivec(1, 2), dvec(1.0, 2.0, 3.0));`

# Serialization

The `mist/serialize.hpp` provides a lightweight, modular serialization framework for writing and reading simulation data. The framework is format-agnostic and supports ASCII, binary, and HDF5 output through a unified interface.

## Design Philosophy

- **Format-agnostic**: Core serialization logic is independent of output format
- **Type-driven**: Uses C++20 concepts to dispatch based on type traits
- **Composable**: Complex types serialize recursively through their components
- **Strict validation**: All fields must be present during deserialization (no optional fields)
- **Consistent with Mist patterns**: Free functions, public underscore-prefixed members

## Core Interface

Two primary entry points:

```cpp
// Serialize object to archive
template<Archive A, typename T>
void serialize(A& archive, const char* name, const T& obj);

// Deserialize object from archive
template<Archive A, typename T>
void deserialize(A& archive, const char* name, T& obj);
```

## Archive Concept

Archives must implement separate reader and writer interfaces:

```cpp
template<typename A>
concept ArchiveWriter = requires(A& ar, const char* name) {
    { ar.write_scalar(name, int{}) } -> std::same_as<void>;
    { ar.write_scalar(name, double{}) } -> std::same_as<void>;
    { ar.write_string(name, std::string{}) } -> std::same_as<void>;
    { ar.write_array(name, vec_t<double, 3>{}) } -> std::same_as<void>;
    { ar.write_array(name, std::vector<double>{}) } -> std::same_as<void>;
    { ar.begin_group(name) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;  // anonymous group
    { ar.end_group() } -> std::same_as<void>;
};

template<typename A>
concept ArchiveReader = requires(A& ar, const char* name, int& i, double& d) {
    { ar.read_scalar(name, i) } -> std::same_as<void>;
    { ar.read_scalar(name, d) } -> std::same_as<void>;
    { ar.read_string(name, std::string&) } -> std::same_as<void>;
    { ar.read_array(name, vec_t<double, 3>&) } -> std::same_as<void>;
    { ar.read_array(name, std::vector<double>&) } -> std::same_as<void>;
    { ar.begin_group(name) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;  // anonymous group
    { ar.end_group() } -> std::same_as<void>;
    { ar.count_groups(name) } -> std::same_as<std::size_t>;
};
```

**Archive implementations:**
- `ascii_writer` / `ascii_reader` - Human-readable text format
- `binary_writer` / `binary_reader` - Compact binary format (planned)
- `hdf5_writer` / `hdf5_reader` - HDF5 hierarchical data format (planned)

## Serializable Types

The framework automatically handles:

1. **Scalars**: `int`, `float`, `double`, and other arithmetic types
2. **Strings**: `std::string` (quoted with escape sequences)
3. **Static vectors**: `vec_t<T, N>` where `T` is arithmetic
4. **Dynamic vectors**: `std::vector<T>` where `T` is serializable
5. **User-defined types**: Any type with `fields()` method

## Enum Serialization

Enums can be serialized as strings by providing ADL `to_string` and `from_string` functions:

```cpp
enum class boundary_condition { periodic, outflow, reflecting };

// ADL to_string: converts enum to string for serialization
inline const char* to_string(boundary_condition bc) {
    switch (bc) {
        case boundary_condition::periodic: return "periodic";
        case boundary_condition::outflow: return "outflow";
        case boundary_condition::reflecting: return "reflecting";
    }
    return "unknown";
}

// ADL from_string: converts string to enum for deserialization
inline boundary_condition from_string(std::type_identity<boundary_condition>, const std::string& s) {
    if (s == "periodic") return boundary_condition::periodic;
    if (s == "outflow") return boundary_condition::outflow;
    if (s == "reflecting") return boundary_condition::reflecting;
    throw std::runtime_error("invalid boundary_condition: " + s);
}
```

With these functions defined, the enum serializes as a quoted string:
```
boundary {
    type = "periodic"
    value = 0.0
}
```

The `HasEnumStrings` concept detects whether a type has these ADL functions available.

## Making Types Serializable

Define both const and non-const versions of `fields()`:

```cpp
struct particle_t {
    vec_t<double, 3> position;
    vec_t<double, 3> velocity;
    double mass;
    
    auto fields() const {
        return std::make_tuple(
            field("position", position),
            field("velocity", velocity),
            field("mass", mass)
        );
    }
    
    auto fields() {
        return std::make_tuple(
            field("position", position),
            field("velocity", velocity),
            field("mass", mass)
        );
    }
};
```

The const version is used for writing, the non-const version for reading.

## ASCII Format Specification

The ASCII archive produces human-readable output with the following formatting rules:

### Formatting Rules

1. **Scalars**: `name = value`
   ```
   time = 1.234
   iteration = 42
   ```

2. **Strings**: `name = "value"` (with escape sequences `\"`, `\\`, `\n`, `\t`, `\r`)
   ```
   title = "Blast Wave Simulation"
   path = "output/data"
   ```

3. **Static vectors** (`vec_t<T, N>`): Inline comma-separated arrays
   ```
   position = [0.1, 0.2, 0.15]
   velocity = [1.5, -0.3, 0.0]
   ```

4. **Dynamic vectors of scalars** (`std::vector<T>` where `T` is arithmetic): Inline comma-separated arrays
   ```
   scalar_field = [300.0, 305.2, 298.5, 302.1]
   ```

5. **Dynamic vectors of compounds** (`std::vector<T>` where `T` is user-defined): Multi-line blocks
   ```
   particles {
       {
           position = [0.1, 0.2, 0.15]
           velocity = [1.5, -0.3, 0.0]
           density = 1.2
           pressure = 101325.0
       }
       {
           position = [0.8, 0.7, 0.25]
           velocity = [-0.5, 0.8, 0.2]
           density = 1.1
           pressure = 98000.0
       }
   }
   ```

6. **Nested structures**: Multi-line with indentation
   ```
   grid {
       resolution = [64, 64, 32]
       domain_min = [0.0, 0.0, 0.0]
       domain_max = [1.0, 1.0, 0.5]
   }
   ```

### Delimiter Rules

- **Commas**: Used inside `[ ]` brackets for array elements
- **Newlines**: Used inside `{ }` braces for fields and blocks
- **No commas**: Between struct blocks or field definitions
- **Indentation**: Each nesting level adds one indentation level

### Complete Example

```cpp
struct grid_config_t {
    vec_t<int, 3> resolution;
    vec_t<double, 3> domain_min;
    vec_t<double, 3> domain_max;
    
    auto fields() const {
        return std::make_tuple(
            field("resolution", resolution),
            field("domain_min", domain_min),
            field("domain_max", domain_max)
        );
    }
    
    auto fields() {
        return std::make_tuple(
            field("resolution", resolution),
            field("domain_min", domain_min),
            field("domain_max", domain_max)
        );
    }
};

struct simulation_state_t {
    double time;
    int iteration;
    grid_config_t grid;
    std::vector<particle_t> particles;
    std::vector<double> scalar_field;

    auto fields() const {
        return std::make_tuple(
            field("time", time),
            field("iteration", iteration),
            field("grid", grid),
            field("particles", particles),
            field("scalar_field", scalar_field)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("time", time),
            field("iteration", iteration),
            field("grid", grid),
            field("particles", particles),
            field("scalar_field", scalar_field)
        );
    }
};

// Serialize to ASCII
simulation_state_t state{...};
ascii_writer ar(std::cout);
serialize(ar, "simulation_state", state);
```

**Output:**
```
simulation_state {
    time = 1.234
    iteration = 42
    grid {
        resolution = [64, 64, 32]
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 0.5]
    }
    particles {
        {
            position = [0.1, 0.2, 0.15]
            velocity = [1.5, -0.3, 0.0]
            density = 1.2
            pressure = 101325.0
        }
        {
            position = [0.8, 0.7, 0.25]
            velocity = [-0.5, 0.8, 0.2]
            density = 1.1
            pressure = 98000.0
        }
    }
    scalar_field = [300.0, 305.2, 298.5, 302.1]
}
```

## Deserialization

Deserialization is strict - all fields defined in `fields()` must be present in the input:

```cpp
simulation_state_t state;
ascii_reader ar(std::ifstream("state.dat"));
deserialize(ar, "simulation_state", state);
// Throws exception if any field is missing
```

**Error handling:**
- Missing field: `Error: Field 'velocity' not found in group 'particles/0'`
- Missing group: `Error: Field 'grid' not found in group 'simulation_state'`

## Usage with Different Archive Types

**ASCII (human-readable):**
```cpp
// Writing
ascii_writer aw(std::ofstream("state.dat"));
serialize(aw, "state", state);

// Reading
ascii_reader ar(std::ifstream("state.dat"));
deserialize(ar, "state", state);
```

**Binary (compact):**
```cpp
// Writing
binary_writer bw(std::ofstream("state.bin", std::ios::binary));
serialize(bw, "state", state);

// Reading
binary_reader br(std::ifstream("state.bin", std::ios::binary));
deserialize(br, "state", state);
```

**HDF5 (hierarchical):**
```cpp
// Writing
hdf5_writer hw("state.h5");
serialize(hw, "state", state);

// Reading
hdf5_reader hr("state.h5");
deserialize(hr, "state", state);
```

All three formats use the same `serialize()` / `deserialize()` interface - only the archive type changes.

# NdArray

The ndarray module provides a unified abstraction for multi-dimensional arrays. The key insight is that an array is fundamentally a mapping from indices to values—not necessarily backed by storage.

## Design

**Lazy arrays** are a pair `(index_space, f)` where `f: index -> value`. No memory is allocated; values are computed on demand. This enables:
- Computed arrays without allocation
- Lazy transformations that compose without materializing intermediates
- Views and slices as index remapping

**Cached arrays** are memory-backed, either owning their buffer or referencing external storage (like `mdspan`).

**Taxonomy:**
```
NdArray
├── lazy_t<S, F>         (space + callable)
└── Cached
    ├── cached_t<T, S>        (owning)
    └── cached_view_t<T, S>   (non-owning)
```

All ndarray types support:
- `space(a)` — the index space
- `start(a)`, `shape(a)`, `size(a)` — delegates to `space(a)`
- `a(idx)` or `at(a, idx)` — element access

## Lazy Arrays

Construct with `lazy(space, func)`:

```cpp
auto space = index_space(ivec(0, 0), uvec(100, 100));

// Coordinate-based values
auto coords = lazy(space, [] MIST_HD (ivec_t<2> idx) {
    return idx[0] + idx[1];
});

// Initial conditions
auto initial = lazy(space, [=] MIST_HD (ivec_t<2> idx) {
    double x = idx[0] * dx;
    double y = idx[1] * dy;
    return exp(-x*x - y*y);
});
```

## Cached Arrays

**Memory locations:**
- `memory::host` — CPU memory
- `memory::device` — GPU memory (cudaMalloc)
- `memory::managed` — Unified memory (cudaMallocManaged)

**Owning vs non-owning:**
- `cached_t` owns its buffer (move-only, no copy)
- `cached_view_t` references external storage

```cpp
// Owning array
cached_t<double, 2> arr(space, memory::host);
arr(ivec(0, 0)) = 1.0;

// View into existing buffer
cached_view_t<double, 2> view(space, ptr);
```

## Operations

### lazy

Construct a lazy array from space and function:

```cpp
auto arr = lazy(space, [] MIST_HD (ivec_t<2> idx) { return ...; });
```

### map

Transform elements lazily:

```cpp
auto doubled = map(arr, [](double x) { return 2.0 * x; });
auto sqrts = map(arr, sqrt);
```

Lvalue sources are captured by pointer; rvalue sources are moved into captures:

```cpp
auto a = map(data, f);              // a references data
auto b = map(std::move(data), f);   // b owns data
```

### cache

Materialize any ndarray to memory:

```cpp
auto host_arr = cache(lazy_arr, memory::host, exec::cpu);
auto device_arr = cache(lazy_arr, memory::device, exec::gpu);
```

For cached sources, uses bulk memcpy:

```cpp
auto device_copy = cache(host_arr, memory::device, exec::cpu);  // h2d
auto host_copy = cache(device_arr, memory::host, exec::cpu);    // d2h
```

### extract

Lazy view into a subregion:

```cpp
auto inner = index_space(ivec(1, 1), uvec(8, 8));
auto sub = extract(arr, inner);
```

### insert

Overlay one array onto another:

```cpp
auto updated = insert(arr, modified_subregion);
```

Indices in `space(modified_subregion)` read from it; others read from `arr`.

## Vector-Valued Arrays

For physics simulations storing `vec_t<T, N>` at each grid point (e.g., conserved variables), use `cached_vec_t` with explicit memory layout:

```cpp
enum class layout { aos, soa };

template<typename T, std::size_t N, std::size_t S, layout L = layout::aos>
struct cached_vec_t;
```

**Layouts:**
- `layout::aos` — Array of Structures: `[v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, ...]`
- `layout::soa` — Structure of Arrays: `[v0.x, v1.x, v2.x, ...], [v0.y, v1.y, v2.y, ...], ...`

SoA provides better GPU memory coalescing. AoS is the default.

**Usage:**

```cpp
auto space = index_space(ivec(0, 0), uvec(100, 100));

// AoS (default) — good for CPU
cached_vec_t<double, 5, 2> cons(space, memory::host);
cons(ivec(0, 0)) = dvec(1.0, 0.0, 0.0, 0.0, 2.5);
vec_t<double, 5> u = cons(ivec(0, 0));

// SoA — good for GPU
cached_vec_t<double, 5, 2, layout::soa> cons_gpu(space, memory::device);
```

**Transparent access:**

Both layouts use the same `operator()` syntax. AoS returns a reference; SoA returns a proxy that gathers on read and scatters on write:

```cpp
template<typename Arr>
void fill_initial(Arr& arr) {
    for (auto idx : space(arr)) {
        arr(idx) = initial_state(idx);  // works for both layouts
    }
}
```

**Caching with layout:**

```cpp
auto device_arr = cache<layout::soa>(lazy_arr, memory::device, exec::gpu);
auto host_arr = cache<layout::aos>(lazy_arr, memory::host, exec::cpu);
```

## Constructors

| Function | Description |
|----------|-------------|
| `lazy(space, f)` | Lazy array from index space and function `f: index -> value` |
| `zeros<T>(space)` | Constant zero |
| `ones<T>(space)` | Constant one |
| `fill<T>(space, value)` | Constant value |
| `indices(space)` | Multi-dimensional index at each position (`ivec_t<S>`) |
| `offsets(space)` | Flat offset at each position (`std::size_t`) |
| `range(n)` | 1D integer sequence `[0, n)` |
| `linspace(start, stop, n)` | 1D evenly spaced values, endpoint inclusive |
| `linspace(start, stop, n, false)` | 1D evenly spaced values, endpoint exclusive |
| `coords(space, origin, delta)` | Physical coordinates (`origin + idx * delta`, returns `dvec_t<S>`) |

## Operators

| Function | Description |
|----------|-------------|
| `map(a, f)` | Lazy element-wise transform `f: value -> value` |
| `extract(a, subspace)` | Lazy view into subregion |
| `insert(a, b)` | Lazy overlay of `b` onto `a` where `space(b) ⊆ space(a)` |
| `cache(a, loc, exec)` | Materialize to memory |

## Typical Patterns

**Transform and materialize:**
```cpp
auto result = cache(map(data, transform), memory::host, exec::cpu);
```

**Extract, transform, insert:**
```cpp
auto inner = index_space(ivec(1, 1), uvec(n-2, n-2));
auto result = cache(
    insert(arr, map(extract(arr, inner), stencil_op)),
    memory::host, exec::cpu
);
```

**GPU computation:**
```cpp
auto device_data = cache(initial_conditions, memory::device, exec::gpu);
auto result = cache(map(device_data, kernel_func), memory::device, exec::gpu);
auto host_result = cache(result, memory::host, exec::cpu);  // d2h
```

## Safe Element Access

`safe_at` handles host/device transfers automatically (expensive, use for debugging):

```cpp
cached_t<double, 2> device_arr(space, memory::device);
safe_at(device_arr, ivec(0, 0)) = 3.14;           // h2d
double val = safe_at(device_arr, ivec(0, 0));     // d2h
```

## Lifetime Management

Lazy arrays from lvalue sources capture by pointer—the source must outlive the lazy array:

```cpp
cached_t<double, 2> data = ...;
auto lazy = map(data, sqrt);  // lazy references data
```

For self-contained lazy arrays, pass rvalues:

```cpp
auto lazy = map(cache(other, memory::host, exec::cpu), sqrt);
// cached_t moved into lazy's captures
```

Convention: lazy arrays are short-lived temporaries, materialized with `cache()` before storage.

## Config Field Setter

The `set()` function allows modifying struct fields by dot-separated path, useful for command-line overrides on restart:

```cpp
// Signature
template<HasFields T>
void set(T& obj, const std::string& path, const std::string& value);

// Usage
config_t config;
set(config, "t_final", "20.0");                    // set scalar
set(config, "physics.gamma", "1.33");              // nested field
set(config, "mesh.boundary.type", "reflecting");   // enum field (uses from_string)
```

Supported target types:
- Arithmetic types: `int`, `long`, `float`, `double`, etc.
- `bool`: accepts "true", "1" for true; anything else is false
- `std::string`: assigned directly
- Enums with `HasEnumStrings`: uses `from_string()` for conversion
- Nested structs with `HasFields`: recursively traverses path

Throws `std::runtime_error` if:
- Field path does not exist
- Target type is not supported
- Enum string conversion fails

## Archive Format Traits

For integration with the driver library, archive formats are defined via trait structs that provide type information and factory functions:

```cpp
struct hdf5_t {
    using reader = hdf5_reader;
    using writer = hdf5_writer;
    
    static constexpr const char* extension = ".h5";
    
    static writer make_writer(const std::string& filename);
    static reader make_reader(const std::string& filename);
};

struct ascii_t {
    using reader = ascii_reader;
    using writer = ascii_writer;

    static constexpr const char* extension = ".dat";

    static writer make_writer(const std::string& filename);
    static reader make_reader(const std::string& filename);
};

struct binary_t {
    using reader = binary_reader;
    using writer = binary_writer;

    static constexpr const char* extension = ".bin";

    static writer make_writer(const std::string& filename);
    static reader make_reader(const std::string& filename);
};
```

These traits allow the driver to be parameterized by archive format:

```cpp
// Driver automatically uses correct file extensions and constructs archives
template<typename Archive, Physics P>
void run(program<P>& prog) {
    // Driver creates: chkpt.0000.h5, chkpt.0001.h5, etc.
}

// User selects format via template parameter
run<hdf5_t>(cfg, state);   // HDF5 format
run<ascii_t>(cfg, state);  // ASCII format
run<binary_t>(cfg, state); // Binary format
```

The trait struct provides:
- **Type aliases**: `reader` and `writer` types for this format
- **File extension**: String literal for output filenames (e.g., ".h5", ".dat", ".bin")
- **Factory functions**: Construct reader/writer instances from filenames

# Driver

The `mist/driver.hpp` provides an interactive time-stepping driver for physics simulations. The driver is a simple REPL (read-eval-print loop) that responds to manual commands for advancing time and generating outputs.

**Current status:** The driver has no notion of scheduled outputs, automated termination conditions, or driver programs. It responds only to manual commands. Future enhancements will add driver programs to encode scheduled outputs and termination logic.

## Physics Concept

Physics modules must satisfy the `Physics` concept by providing:

**Required types:**
- `config_t` - Runtime configuration including `rk_order` and `cfl`. Must implement `fields()`.
- `initial_t` - Initial condition parameters. Must implement `fields()`.
- `state_t` - Conservative state variables. Must implement `fields()`.
- `product_t` - Derived diagnostic quantities.
- `exec_context_t` - Execution context (combines config, initial, and any runtime data).

**Required functions:**
- `default_physics_config(std::type_identity<P>) -> config_t` - Default physics configuration
- `default_initial_config(std::type_identity<P>) -> initial_t` - Default initial configuration
- `initial_state(exec_context_t) -> state_t` - Generate initial conditions
- `courant_time(state_t, exec_context_t) -> double` - CFL-limited timestep
- `zone_count(state_t, exec_context_t) -> size_t` - Number of computational zones
- `names_of_time(std::type_identity<P>) -> vector<string>` - Available time variable names
- `names_of_timeseries(std::type_identity<P>) -> vector<string>` - Available timeseries columns
- `names_of_products(std::type_identity<P>) -> vector<string>` - Available product names
- `get_time(state_t, string) -> double` - Get a time variable by name
- `get_timeseries(config_t, initial_t, state_t, string) -> double` - Get a timeseries value by name
- `get_product(state_t, string, exec_context_t) -> product_t` - Get a product by name
- `advance(state_t&, dt, exec_context_t) -> void` - Advance state by timestep dt

## Program Structure

```cpp
template<Physics P>
struct program_t {
    typename P::config_t physics;                  // Physics configuration
    typename P::initial_t initial;                 // Initial condition parameters
    std::optional<typename P::state_t> physics_state;  // Physics state (null until init)
    driver::state_t driver_state;                  // Driver state
};
```

The driver maintains its own state (output format, file counters, timeseries data, repeating commands). Physics parameters live in the physics module.

## Time Integration

Time integration is handled by the physics module's `advance()` function. The physics module is responsible for implementing its own time-stepping scheme (e.g., SSP Runge-Kutta methods). The `mist/runge_kutta.hpp` header provides utilities for implementing RK methods:

- Order 1: Forward Euler
- Order 2: SSP-RK2
- Order 3: SSP-RK3

## Interactive Commands

The driver provides a GNU readline-enabled REPL with the following commands:

**Stepping:**
- `n++` - Advance 1 iteration
- `n += 10` - Advance 10 iterations
- `n -> 1000` - Advance to iteration 1000
- `t += 10.0` - Advance time by exactly 10.0
- `t -> 20.0` - Advance to exactly time 20.0
- `orbit += 3.0` - Advance until orbit increases by at least 3.0
- `orbit -> 60.0` - Advance until orbit reaches at least 60.0

**Configuration:**
- `set output=ascii` - Set output format (ascii|binary|hdf5)
- `set physics key=val ...` - Set physics config parameters
- `set initial key=val ...` - Set initial data parameters (only when state is null)
- `select products [prod1 ...]` - Select products (no args = all)
- `select timeseries [col1 ...]` - Select timeseries columns (no args = all)
- `clear timeseries` - Clear timeseries data

**State management:**
- `init` - Generate initial state from config
- `reset` - Reset driver and clear physics state
- `load <file>` - Load configuration data or a checkpoint file

**Sampling:**
- `do timeseries` - Record timeseries sample

**File I/O:**
- `write timeseries [file]` - Write timeseries to file
- `write checkpoint [file]` - Write checkpoint to file
- `write products [file]` - Write products to file
- `write message <text>` - Write custom message to stdout

**Recurring commands:**
- `repeat <interval> <unit> <cmd>` - Execute command every interval ('do' or 'write')
- `repeat list` - List active recurring commands
- `repeat clear` - Clear all recurring commands

**Information:**
- `show all` - Show all state information
- `show physics` - Show physics configuration
- `show initial` - Show initial configuration
- `show products` - Show available and selected products
- `show timeseries` - Show timeseries data
- `show driver` - Show driver state (including repeating tasks)
- `help` - Show this help
- `stop | quit | q` - Exit simulation

## Example Session

```
$ ./advection-1d-decomp

Type 'help' for available commands
> show physics
physics {
    rk_order = 2
    cfl = 0.4
    num_zones = 200
    domain_length = 1.0
    advection_velocity = 1.0
}
> init
[000000] t=0.00000
> select timeseries t x_com momentum
> n += 5
[000005] t=0.0100 Mzps=3.465
> do timeseries
> t -> 1.0
[000500] t=1.0000 Mzps=2.954
> do timeseries
> write timeseries data.dat
Wrote data.dat
> write checkpoint
Wrote checkpoint.0000.dat
> stop
```

## Output Files

**Checkpoints:** `chkpt.NNNN.dat` - Full program state including:
- Physics configuration
- Physics state
- Iteration counter
- Driver state (output format, file counters, timeseries data)

**Products:** `prods.NNNN.dat` - Derived quantities from `get_product()`

**Timeseries:** User-specified file (e.g., `data.txt` or `data.bin`) - Uses serialization framework:
- Respects `set output=ascii|binary` format setting
- ASCII format: nested structure with columns list and data map
- Binary format: compact binary encoding

Checkpoint and products file numbering increments with each manual output command (not tied to iteration count).

## Timeseries Workflow

1. **Select columns**: `select timeseries col1 col2 col3`
   - Specifies which physics-provided diagnostics to track
   - Columns must exist in `timeseries_sample()` output
   - Column ordering in output is alphabetical (std::map ordering)

2. **Record samples**: `do timeseries`
   - Calls `timeseries_sample(config, state)` from physics module
   - Extracts only the selected columns
   - Appends values to the timeseries data structure
   - Throws error if selected column is missing from physics output

3. **Write to file**: `write timeseries filename.dat`
   - Uses the serialization framework (respects output format setting)
   - Can be called multiple times with different filenames

4. **Clear data**: `clear timeseries`
   - Clears accumulated timeseries data
   - Use before `select timeseries` to start fresh

5. **Persistence**: Timeseries data is saved in checkpoints
   - On restart, timeseries accumulation continues from checkpoint state
   - Use `clear timeseries` then `select timeseries` after restart to begin fresh

## Example Physics Module

```cpp
struct my_physics {
    struct config_t {
        int rk_order = 2;
        double cfl = 0.4;
        // ... physics-specific parameters

        auto fields() const { return std::make_tuple(
            field("rk_order", rk_order),
            field("cfl", cfl),
            // ... other fields
        );}

        auto fields() { /* same */ }
    };

    struct state_t {
        std::vector<double> conserved;
        double time;

        auto fields() const { return std::make_tuple(
            field("conserved", conserved),
            field("time", time)
        );}

        auto fields() { /* same */ }
    };

    struct product_t {
        // ... derived quantities
        auto fields() const { /* ... */ }
        auto fields() { /* ... */ }
    };
};

// Required free functions
auto default_config(std::type_identity<my_physics>) -> my_physics::config_t;
auto initial_state(const my_physics::config_t&, double t) -> my_physics::state_t;
auto courant_time(const my_physics::config_t&, const my_physics::state_t&) -> double;
void rk_step(const my_physics::config_t&, my_physics::state_t&,
             const my_physics::state_t&, double dt, double alpha);
auto copy(const my_physics::state_t&) -> my_physics::state_t;
auto get_product(const my_physics::config_t&, const my_physics::state_t&) -> my_physics::product_t;
auto get_named_times(const my_physics::state_t&) -> std::vector<std::pair<std::string, double>>;
auto zone_count(const my_physics::state_t&) -> std::size_t;
auto timeseries_sample(const my_physics::config_t&, const my_physics::state_t&)
    -> std::vector<std::pair<std::string, double>>;
```

## Main Function

```cpp
int main() {
    mist::program_t<my_physics> prog;
    prog.physics = default_physics_config(std::type_identity<my_physics>{});
    prog.initial = default_initial_config(std::type_identity<my_physics>{});

    auto final_state = mist::run(prog);
    return 0;
}
```

## Future: Driver Programs

The current driver responds only to manual commands. A future enhancement will add driver programs that encode:
- Target times and termination conditions
- Scheduled output policies (messages, checkpoints, products, timeseries)
- Restart behavior

This will enable automated runs while preserving the current interactive mode for debugging.
