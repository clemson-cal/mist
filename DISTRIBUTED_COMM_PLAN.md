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

## Next Steps

### 1. MPI Backend for comm_t

```cpp
struct comm_t {
    void* mpi_comm_ = nullptr;  // MPI_Comm when MPI enabled

    // Factory
    static auto from_mpi(MPI_Comm comm) -> comm_t;
};
```

**build_plan with MPI:**
- Allgather publication metadata (spaces only) across ranks
- For each local request, find remote providers
- Populate `sends` and `recvs` vectors in the plan

**exchange with MPI:**
- Build MPI subarray datatypes from views (using parent space)
- Post MPI_Irecv for each recv
- Post MPI_Isend for each send
- MPI_Waitall
- Tag generation: hash of (src_rank, dest_rank, overlap)

**combine with MPI:**
- MPI_Allreduce with the provided binary op
- May need to register custom MPI_Op or use predefined ops

### 2. Plan Caching in ExchangeStage

```cpp
struct some_exchange_stage {
    mutable std::optional<exchange_plan_t<SrcView, DestView>> cached_plan_;
    mutable std::size_t topology_version_ = 0;

    auto plan(contexts, comm) -> const exchange_plan_t<...>&;
};
```

- Cache invalidation when topology changes (AMR, load balancing)
- Could use version counter or hash of index spaces

### 3. MPI Datatype Construction

```cpp
// From a view, build MPI subarray type:
auto make_mpi_type(const View& view) -> MPI_Datatype {
    MPI_Type_create_subarray(
        ndims,
        parent(view).shape(),           // size of parent array
        space(view).shape(),            // size of subregion
        space(view).start() - parent(view).start(),  // offset
        MPI_ORDER_C,
        element_mpi_type<View::value_type>(),
        &subarray_type
    );
    MPI_Type_commit(&subarray_type);
    return subarray_type;
}
```

### 4. Testing

- Multi-rank MPI tests (need MPI environment)
- Integration tests with pipelines across ranks
- Verify ghost exchange correctness with multiple patches per rank

### 5. Optional Enhancements

- **Persistent communication**: reuse MPI requests across iterations
- **Overlapping communication with computation**: for async pipelines
- **GPU-aware MPI**: handle device memory in transfers
