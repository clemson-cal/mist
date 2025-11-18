# Overview
Mist is a very lightweight C++20 header-only library for deploy-anywhere array transformations.

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
