# Distributed Communication Implementation Plan

## Completed

1. **array_view_t stores parent space** (not strides)
   - Enables MPI_Type_create_subarray construction
   - Added `parent()` free function

2. **ExchangeStage::provides() returns array_view_t**
   - Removed redundant `data()` method and `space_t` typedef
   - Simpler concept

3. **ReduceStage with extract/combine/finalize**
   - `init()`, `combine()` are static (value operations)
   - `extract()`, `finalize()` are instance methods (context access)
   - `combine` works on two values - enables distributed reduction

4. **comm_t integrated with pipeline.hpp**
   - `exchange_plan_t<SrcView, DestView>` supports separate types
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
   - `detail::make_mpi_subarray()` creates MPI subarray types from views
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

## Next Steps

### 1. Plan Caching in ExchangeStage

```cpp
struct some_exchange_stage {
    mutable std::optional<exchange_plan_t<SrcView, DestView>> cached_plan_;
    mutable std::size_t topology_version_ = 0;

    auto plan(contexts, comm) -> const exchange_plan_t<...>&;
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
