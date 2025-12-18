# Core Library

The core library (`mist/core.hpp`) provides multi-dimensional arrays, index spaces, and parallel traversals.

## Data Structures

All data structures use overloaded free functions rather than member functions. Data members are public but prefixed with underscore.

### vec_t

`vec_t<T, S>` is a statically sized array with element type `T` and size `S`.

**Aliases:**
- `dvec_t<S>` — `vec_t<double, S>`
- `ivec_t<S>` — `vec_t<int, S>`
- `uvec_t<S>` — `vec_t<unsigned int, S>`

**Constructors:**
```cpp
vec(0.4, 1.0, 2.0)   // -> vec_t<double, 3>, type deduced via std::common_type
dvec(1.0, 2.0)       // -> dvec_t<2>
ivec(0, 0)           // -> ivec_t<2>
uvec(10, 20)         // -> uvec_t<2>
range<3>()           // -> uvec_t<3>{0, 1, 2}
```

**Operators:**
- `vec_t<T, S> + vec_t<U, S>`, `vec_t<T, S> - vec_t<U, S>`
- `vec_t<T, S> * U`, `U * vec_t<T, S>`, `vec_t<T, S> / U`
- `==`

**Free functions:**
- `dot(a, b)` — dot product
- `map(v, f)` — element-wise transformation
- `sum(v)` — sum of elements
- `product(v)` — product of elements
- `any(v)` — true if any element is true (for `vec_t<bool, S>`)
- `all(v)` — true if all elements are true

### index_space_t

`index_space_t<S>` defines an S-dimensional rectangular region via `_start: ivec_t<S>` and `_shape: uvec_t<S>`.

All indices are **absolute** (relative to origin `[0, 0, ...]`). With `_start = [5, 10]` and `_shape = [10, 20]`, valid indices range from `[5, 10]` to `[14, 29]`.

**Construction:**
```cpp
auto space = index_space(ivec(0, 0), uvec(10, 20));
```

**Free functions:**
- `start(space)` — returns `_start`
- `shape(space)` — returns `_shape`
- `size(space)` — total element count
- `upper(space)` — `_start + _shape` (one past the end)
- `contains(space, index)` — bounds check

**Iteration:**
```cpp
for (auto idx : space) { ... }  // iterates all indices
```

**Partitioning:**
```cpp
subspace(space, n, i, axis)  // partition into n parts, return part i along axis
```

**Modifications:**
- `shift(space, offset)` — translate start
- `expand(space, amount)` / `contract(space, amount)` — adjust shape symmetrically
- `nudge(space, lo, hi)` — adjust start by `lo`, shape by `hi - lo`

## Multi-Dimensional Indexing

Convert between multi-dimensional indices and flat buffer offsets (row-major ordering).

```cpp
ndoffset(space, index)        // ivec_t<S> -> std::size_t
ndindex(space, offset)        // std::size_t -> ivec_t<S>
ndread(data, space, index)    // read T from buffer
ndwrite(data, space, index, value)  // write T to buffer
```

**SoA (Struct of Arrays) layout** for vector fields:
```cpp
ndread_soa<T, N>(data, space, index)   // read vec_t<T, N>
ndwrite_soa<T, N>(data, space, index, value)  // write vec_t<T, N>
```

Component `i` of vector at `index` is at offset `i * size(space) + ndoffset(space, index)`.

## Array Traversal

Traverse index spaces with optional parallel execution.

**Execution policies:**
- `exec::cpu` — sequential
- `exec::omp` — OpenMP parallel
- `exec::gpu` — CUDA kernel

### for_each

Apply a function to each index:
```cpp
for_each(space, [&](ivec_t<2> idx) {
    arr(idx) = compute(idx);
}, exec::cpu);
```

### map_reduce

Parallel map followed by reduction:
```cpp
auto total = map_reduce(space, 0.0,
    [&](ivec_t<2> idx) { return arr(idx); },  // map: index -> value
    std::plus<>{},                             // reduce: combine values
    exec::omp
);
```

---

# NdArray

The ndarray module (`mist/ndarray.hpp`) provides lazy and cached array abstractions.

## Design

An array is a mapping from indices to values—not necessarily memory-backed.

**Lazy arrays** are pairs `(index_space, f)` where `f: index -> value`. No allocation; values computed on demand.

**Cached arrays** are memory-backed, either owning their buffer or referencing external storage.

**Taxonomy:**
```
NdArray
├── lazy_t<S, F>              (space + callable)
└── Cached
    ├── cached_t<T, S>        (owning)
    └── cached_view_t<T, S>   (non-owning)
```

All types support:
- `space(a)` — the index space
- `start(a)`, `shape(a)`, `size(a)` — delegates to space
- `a(idx)` or `at(a, idx)` — element access

## Lazy Arrays

Construct with `lazy(space, func)`:
```cpp
auto coords = lazy(space, [](ivec_t<2> idx) {
    return idx[0] + idx[1];
});
```

Lvalue sources captured by pointer; rvalue sources moved into captures.

## Cached Arrays

**Memory locations:**
- `memory::host` — CPU
- `memory::device` — GPU (cudaMalloc)
- `memory::managed` — unified memory

**Types:**
- `cached_t<T, S>` — owns buffer (move-only)
- `cached_view_t<T, S>` — references external storage

```cpp
cached_t<double, 2> arr(space, memory::host);
arr(ivec(0, 0)) = 1.0;
```

## Vector-Valued Arrays

`cached_vec_t<T, N, S, L>` stores `vec_t<T, N>` at each grid point.

**Layouts:**
- `layout::aos` — `[v0.x, v0.y, v0.z, v1.x, ...]` (CPU friendly)
- `layout::soa` — `[v0.x, v1.x, ...], [v0.y, v1.y, ...]` (GPU coalescing)

```cpp
cached_vec_t<double, 5, 2> cons(space, memory::host);           // AoS
cached_vec_t<double, 5, 2, layout::soa> cons_gpu(space, memory::device);  // SoA
```

Both layouts use same `operator()` syntax.

## Operations

| Function | Description |
|----------|-------------|
| `lazy(space, f)` | Lazy array from function |
| `map(a, f)` | Lazy element-wise transform |
| `extract(a, subspace)` | Lazy view into subregion |
| `insert(a, b)` | Lazy overlay of b onto a |
| `cache(a, loc, exec)` | Materialize to memory |

## Constructors

| Function | Description |
|----------|-------------|
| `zeros<T>(space)` | Constant zero |
| `ones<T>(space)` | Constant one |
| `fill<T>(space, value)` | Constant value |
| `indices(space)` | Multi-dim index at each position |
| `offsets(space)` | Flat offset at each position |
| `range(n)` | 1D sequence `[0, n)` |
| `linspace(start, stop, n)` | Evenly spaced, endpoint inclusive |
| `coords(space, origin, delta)` | Physical coordinates |

## Patterns

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
auto device_data = cache(initial, memory::device, exec::gpu);
auto result = cache(map(device_data, kernel), memory::device, exec::gpu);
auto host_result = cache(result, memory::host, exec::cpu);  // d2h
```

---

# Serialization

The serialization framework (`mist/serialize.hpp`) provides format-agnostic reading and writing.

## Interface

```cpp
serialize(archive, "name", obj);    // write
deserialize(archive, "name", obj);  // read
```

## Supported Types

- Scalars: `int`, `double`, etc.
- Strings: `std::string`
- Static vectors: `vec_t<T, N>`
- Dynamic vectors: `std::vector<T>`
- User types with `fields()` method

## Making Types Serializable

Define both const and non-const `fields()`:
```cpp
struct particle_t {
    vec_t<double, 3> position;
    double mass;

    auto fields() const {
        return std::make_tuple(
            field("position", position),
            field("mass", mass)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("position", position),
            field("mass", mass)
        );
    }
};
```

## Enum Serialization

Provide ADL `to_string` and `from_string`:
```cpp
enum class boundary { periodic, outflow };

const char* to_string(boundary b) {
    switch (b) {
        case boundary::periodic: return "periodic";
        case boundary::outflow: return "outflow";
    }
    return "unknown";
}

boundary from_string(std::type_identity<boundary>, const std::string& s) {
    if (s == "periodic") return boundary::periodic;
    if (s == "outflow") return boundary::outflow;
    throw std::runtime_error("invalid boundary: " + s);
}
```

## Archive Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `ascii_t` | `.dat` | Human-readable text |
| `binary_t` | `.bin` | Compact binary |
| `hdf5_t` | `.h5` | HDF5 hierarchical |

All use the same `serialize()`/`deserialize()` interface.

## Config Field Setter

Modify struct fields by dot-separated path:
```cpp
set(config, "physics.gamma", "1.33");
set(config, "mesh.boundary", "reflecting");
```
