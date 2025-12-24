# Distributed Communication Implementation Plan

## Completed

1. **array_view_t stores parent space** (not strides)
   - Enables MPI_Type_create_subarray construction
   - Added `parent()` free function

2. **ExchangeStage concept with value_type and rank**
   - Stage defines `value_type` (element type) and `rank` (dimensionality)
   - View types derived as `array_view_t<const T, S>` and `array_view_t<T, S>`
   - Removed `buffer_t` typedef in favor of explicit types

3. **ReduceStage with extract/combine/finalize**
   - `init()`, `combine()` are static (value operations)
   - `extract()`, `finalize()` are instance methods (context access)
   - `combine` works on two values - enables distributed reduction

4. **comm_t integrated with pipeline.hpp**
   - `exchange_plan_t<T, S>` templated on value type and rank
   - `execute_exchange()` uses `comm.build_plan()` + `comm.exchange()`
   - `execute_reduce()` uses `comm.combine()` for global reduction
   - Backwards-compatible overloads with local-only communicator

5. **MPI Backend for comm_t**
   - Factory method: `comm_t::from_mpi(MPI_Comm)`
   - Stores `MPI_Comm` when `MIST_WITH_MPI` is defined
   - `build_plan()` uses MPI_Allgatherv to exchange publication/request metadata
   - `exchange()` uses MPI_Isend/MPI_Irecv/MPI_Waitall for non-blocking transfers
   - `combine()` uses MPI_Allgather + local reduction with custom binary op
   - Tags generated from hash of overlap region

6. **MPI Datatype Construction**
   - `detail::make_mpi_subarray<T, S>()` creates MPI subarray types
   - Uses parent space for proper strided access
   - Supports `double`, `float`, `int`, and `vec_t<T, N>` element types
   - `detail::mpi_type_traits<T>` for type mapping

7. **CMake Integration**
   - `MIST_WITH_MPI` option enables MPI support
   - Automatically finds MPI and links `MPI::MPI_CXX`
   - Defines `MIST_WITH_MPI` preprocessor macro

8. **MPI Tests**
   - `test_comm_mpi.cpp` with tests for:
     - `comm_t::from_mpi()` factory
     - `combine()` with sum and min operations
     - `exchange()` for 1D ghost cell communication
   - Runs with `mpirun -np 4` in CTest

9. **Simplified pipeline.hpp**
   - Removed unused includes (`<algorithm>`, `<atomic>`, `<functional>`)
   - Removed unused `pipeline_t::get()` method
   - Merged detail namespace blocks
   - Simplified context type using `pipeline_t<...>::context_t`

10. **laplacian2d Example with Distributed Domain Decomposition**
    - Exchange-compute-reduce pipeline without driver framework
    - Ghost zone exchange between 2D patches using `ghost_exchange_t` stage
    - 5-point Laplacian stencil computation via `compute_laplacian_t`
    - L2 error reduction using `error_reduce_t` ReduceStage pattern
    - Uses `subspace()` and `ndindex()` for domain decomposition
    - Demonstrates multi-patch per rank distributed execution

11. **core.hpp Cleanup and Refactoring**
    - Consolidated `ndindex()` with single C-ordering (last index fastest)
    - Removed `unravel()` function (use `ndindex(offset, shape)` instead)
    - Added `to_signed()` helper for uvec_t → ivec_t conversion
    - Added unary minus operator for vec_t
    - Added vec_t static methods: `constant()`, `zeros()`, `ones()`
    - Replaced `static_cast<T>(x)` with functional casts `T(x)` throughout
    - Used `[]` accessor instead of `._data[]` in index_space functions
    - Shortened variable names: `space→s`, `index→i/idx`, `offset→off`, `shape→sh`, etc.
    - Used `auto` declarations for function return types and obvious types
    - Removed underscore prefix from member variables: `_data→data`, `_start→start`, `_shape→shape`

12. **Header File Member Variable Cleanup**
    - Removed underscore prefix from all member variables across headers
    - vec_t: `_data → data`
    - index_space_t: `_start → start`, `_shape → shape`
    - index_space_iterator: `_space → space`, `_offset → offset`
    - array_t: `_space → space`, `_data → data`, `_location → location`
    - lazy_t: `_space → space`, `_func → func`
    - array_view_t: `_parent → parent`
    - Fixed typo: `indexspace_t → index_space_t`, `indexspace() → index_space()`

13. **Simulation DSL for Low-Boilerplate Parallel Execution**
    - Four increasingly sophisticated approaches in laplacian2d.cpp:
      * `main()` - Educational detail with explicit setup (68 lines)
      * `main_a()` - Generic factory helpers (39 lines, ~30 lines saved)
      * `main_b()` - Minimal lambda-based (45 lines, modern C++)
      * `main_d()` - General DSL with products (55 lines, driver-friendly)
    - `mpi_context` RAII class: Encapsulates MPI init/finalize, no more #ifdef clutter
    - `simulation<PatchType>` DSL class with:
      * `decompose_cartesian()` - Regular grid decomposition
      * `decompose_custom()` - User-defined decomposition for AMR
      * `.execute(exchange, compute, reduce)` - Pipeline execution
      * `.define_product(name, extractor)` - Named outputs for driver
      * `.run()` - Returns `unordered_map<string, any>` of products
    - **Genericity**: Template on PatchType enables 1D/2D/3D/AMR without code changes
    - **Driver Integration**: Products pattern avoids global state
    - **Documentation**: DSL_DESIGN.md with usage examples for wave, hydro, AMR

## Next Steps

### 1. Plan Caching in ExchangeStage

```cpp
struct some_exchange_stage {
    mutable std::optional<exchange_plan_t<T, S>> cached_plan_;
    mutable std::size_t topology_version_ = 0;

    auto plan(contexts, comm) -> const exchange_plan_t<T, S>&;
};
```

- Cache invalidation when topology changes (AMR, load balancing)
- Could use version counter or hash of index spaces

### 2. Integration Testing

- Multi-rank pipeline execution with actual simulation codes
- Verify ghost exchange correctness with multiple patches per rank
- Test with advect1d and srhd1d examples using MPI

### 3. Optional Enhancements

- **Persistent communication**: reuse MPI requests across iterations
- **Overlapping communication with computation**: for async pipelines
- **GPU-aware MPI**: handle device memory in transfers
