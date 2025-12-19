# Distributed Mode Design

This document describes the design for massively parallel / distributed execution in mist.

## Goals

- No changes to physics modules for distribution
- Only `main()` changes to establish communications
- Generic abstraction layer — no MPI exposure to user code
- Support both uniform grids and AMR

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Rank 0                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   REPL   │───▶│  parse   │───▶│ command  │──────────────┼──┐
│  └──────────┘    └──────────┘    └──────────┘              │  │
│       ▲                                                     │  │
│       │                          ┌──────────┐              │  │
│       └──────────────────────────│ response │◀─────────────┼──┼──┐
│                                  └──────────┘              │  │  │
├─────────────────────────────────────────────────────────────┤  │  │
│ All Ranks                                      broadcast   │  │  │
│                                                    ┌───────┼──┘  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     ▼       │     │
│  │ engine_t │───▶│ execute  │───▶│ response │─────────────┼─────┘
│  │ (local)  │    │ (local)  │    │ (local)  │   reduce    │
│  └──────────┘    └──────────┘    └──────────┘             │
└─────────────────────────────────────────────────────────────┘
```

Each rank has its own `engine_t` operating on local patches. Commands are broadcast so all engines execute the same command. Responses are reduced/gathered as needed.

## Three Abstraction Layers

### 1. Communicator

Moves bytes between ranks. Knows nothing about grids or patches.

```cpp
struct communicator_t {
    virtual int rank() const = 0;
    virtual int size() const = 0;
    bool is_root() const { return rank() == 0; }

    // Point-to-point (for halo exchange)
    virtual void sendrecv(int peer,
                          std::span<const std::byte> send,
                          std::span<std::byte> recv) = 0;

    // Collective reductions
    virtual double allreduce_min(double) = 0;
    virtual double allreduce_sum(double) = 0;

    // Command distribution
    virtual void broadcast_command(command_t&) = 0;

    virtual void barrier() = 0;
    virtual ~communicator_t() = default;
};
```

Implementations:
- `null_communicator_t` — single process, all operations are no-ops
- `mpi_communicator_t` — wraps MPI calls (compiled conditionally)

### 2. Grid Topology

Describes patch structure, levels, and neighbor relationships. Pure geometry, no knowledge of ranks.

```cpp
template<int Rank>
struct grid_t {
    std::vector<patch_descriptor_t> patches;

    auto neighbors(int patch_id) const -> std::vector<neighbor_info_t>;
    auto level(int patch_id) const -> int;
    auto space(int patch_id) const -> index_space_t<Rank>;
};

struct patch_descriptor_t {
    int id;
    int level;                    // 0 for unigrid, varies for AMR
    index_space_t<Rank> space;    // global coords at this level
};

struct neighbor_info_t {
    int patch_id;
    index_space_t<Rank> overlap;  // region to exchange
    int level_delta;              // 0 = same level, +1 = finer, -1 = coarser
};
```

### 3. Distribution

Maps patches to ranks. Knows grid structure, assigns ownership.

```cpp
struct distribution_t {
    virtual auto rank_for_patch(int patch_id) const -> int = 0;
    virtual auto patches_for_rank(int rank) const -> std::vector<int> = 0;
};
```

Implementations:
- `round_robin_distribution_t` — patch i goes to rank i % size
- `spatial_distribution_t` — nearby patches on same rank
- `weighted_distribution_t` — load-balanced by patch cost

## Grid Topologies

### Uniform Grid (Domain Decomposition)

All patches have the same size and refinement level:

```
┌─────┬─────┬─────┐
│  0  │  1  │  2  │
├─────┼─────┼─────┤
│  3  │  4  │  5  │
└─────┴─────┴─────┘

All level 0, all same size
Simple halo exchange between neighbors
```

### AMR Grid

Patches at varying refinement levels covering a global finest-resolution index space:

```
┌───────────┬─────┬─────┐
│           │  2  │  3  │  level 1 (fine)
│     0     ├─────┼─────┤
│           │  4  │  5  │
├───────────┴─────┴─────┤
│           1           │  level 0 (coarse)
└───────────────────────┘
```

Cross-level neighbors require prolongation (coarse→fine) or restriction (fine→coarse):

```cpp
struct amr_exchange_policy {
    void prolongate(std::span<const double> coarse,
                    std::span<double> fine,
                    index_space_t<Rank> region);

    void restrict(std::span<const double> fine,
                  std::span<double> coarse,
                  index_space_t<Rank> region);
};
```

## SPMD Execution Model

All ranks run the same code. Each rank:
1. Computes its patch assignment from global patch list + rank
2. Calls `initial_state()` for its own patches only (no scatter)
3. Exchanges halos with neighbor ranks
4. Contributes to global reductions

```cpp
// In exec_context_t or similar
auto my_patches() const {
    return dist.patches_for_rank(comm.rank());
}

auto rank_for_patch(int patch_id) const {
    return dist.rank_for_patch(patch_id);
}
```

## Distributed Session

A new session type handles command distribution. The engine stays local-only.

```cpp
struct distributed_session_t {
    engine_t& engine;
    communicator_t& comm;

    void run() {
        while (true) {
            command_t cmd;

            // Only rank 0 reads input
            if (comm.is_root()) {
                cmd = read_command();
            }

            // All ranks receive same command
            comm.broadcast_command(cmd);

            if (std::holds_alternative<stop_t>(cmd)) {
                break;
            }

            // All ranks execute on their local patches
            auto local_response = engine.execute(cmd);

            // Combine responses if needed
            auto response = reduce_response(local_response, comm);

            // Rank 0 reports result
            if (comm.is_root()) {
                write_response(response);
            }
        }
    }
};
```

## Response Aggregation

Different commands require different response handling:

| Command | Local Execution | Response Aggregation |
|---------|-----------------|----------------------|
| `init` | each rank inits its patches | rank 0 reports ok |
| `advance_to` | all advance in lockstep | iteration/time from any rank |
| `write_checkpoint` | each rank writes its patches | rank 0 reports filenames |
| `show_profiler` | each has local timing | allreduce (sum/max) |
| `get_products` | each has local data | gather or parallel I/O |

## Pipeline Execution

The pipeline execution function receives distributed execution context:

```cpp
void execute(std::vector<context_t>& contexts,
             Pipeline pipeline,
             scheduler_t& sched,
             communicator_t* comm,
             const distribution_t* dist,
             const grid_t<Rank>* grid,
             profiler_t* prof);
```

During `ExchangeStage`, the pipeline determines local vs remote:

```cpp
for (auto& neighbor : grid->neighbors(patch_id)) {
    int provider_rank = dist->rank_for_patch(neighbor.patch_id);

    if (provider_rank == comm->rank()) {
        // Local: memcpy from local context
    } else {
        // Remote: comm->sendrecv(provider_rank, ...)
    }
}
```

## User Code Changes

### Before (single process)

```cpp
int main() {
    auto physics = mist::driver::make_physics<my_physics>();
    auto state = mist::driver::state_t{};
    auto engine = mist::driver::engine_t{state, *physics};
    auto session = mist::driver::repl_session_t{engine};
    session.run();
}
```

### After (distributed)

```cpp
int main(int argc, char** argv) {
    auto comm = mist::driver::make_communicator(argc, argv);
    auto physics = mist::driver::make_physics<my_physics>();
    auto state = mist::driver::state_t{};
    auto engine = mist::driver::engine_t{state, *physics};
    auto session = mist::driver::distributed_session_t{engine, *comm};
    session.run();
}
```

The physics module requires no changes.

## Build Integration

```cmake
option(MIST_USE_MPI "Enable MPI support" OFF)

if(MIST_USE_MPI)
    find_package(MPI REQUIRED)
    target_sources(mist_driver PRIVATE src/driver/mpi_communicator.cpp)
    target_link_libraries(mist_driver PUBLIC MPI::MPI_CXX)
    target_compile_definitions(mist_driver PUBLIC MIST_HAS_MPI)
endif()
```

## Distributed Output

Simplest approach: each rank writes its own files, rank 0 handles interactive I/O.

### Checkpoints

Each rank writes its own patches to a separate file:

```
checkpoint.0000.rank0.dat
checkpoint.0000.rank1.dat
checkpoint.0000.rank2.dat
```

On restart, each rank reads patches it owns. If restarting with a different rank count, each rank reads from whichever old files contain its patches:

```cpp
void load_checkpoint(int checkpoint_num) {
    for (int patch_id : my_patches()) {
        int old_rank = read_patch_location(checkpoint_num, patch_id);
        load_patch(checkpoint_num, old_rank, patch_id);
    }
}
```

### Products

Each rank writes its own patches:

```
products.0000.rank0.h5
products.0000.rank1.h5
```

Post-processing or visualization tools stitch patches together as needed.

### Timeseries

Root-only. Global quantities (total energy, max velocity, etc.) are computed via reduction, and only rank 0 records and writes them:

```cpp
// During advance
double local_max_v = compute_local_max_velocity();
double global_max_v = comm.allreduce_max(local_max_v);

if (comm.is_root()) {
    state.timeseries["max_velocity"].push_back(global_max_v);
}
```

### Python / TUI

Root gathers on demand. When Python requests product data:

1. Request arrives at rank 0 via socket
2. Rank 0 broadcasts "gather products" to all ranks
3. Each rank sends its data to rank 0
4. Rank 0 assembles and sends to Python

This is simple and sufficient for interactive use. For large-scale visualization, read the per-rank product files directly.

## Summary

| Object | Responsibility | Knows About |
|--------|----------------|-------------|
| `communicator_t` | move bytes between ranks | ranks only |
| `grid_t` | patch structure, neighbors | geometry only |
| `distribution_t` | patch → rank mapping | grid + ranks |
| `distributed_session_t` | command broadcast, response reduce | all three |
| `engine_t` | execute commands locally | local patches only |
| physics module | define physics | none of these |

## Implementation Status

**Files created:**
- `include/mist/driver/communicator.hpp` — abstract interface + `null_communicator_t`
- `include/mist/driver/distributed_session.hpp` — distributed session header
- `src/driver/distributed_session.cpp` — implementation

**Communicator interface (current):**
```cpp
struct communicator_t {
    virtual int rank() const = 0;
    virtual int size() const = 0;
    bool is_root() const;

    virtual void sendrecv(int peer, span<const byte> send, span<byte> recv) = 0;
    virtual double allreduce_min(double) = 0;
    virtual double allreduce_max(double) = 0;
    virtual double allreduce_sum(double) = 0;
    virtual void broadcast_string(std::string&) = 0;
    virtual void barrier() = 0;
};
```

**MPI support (complete):**
- `mpi_communicator_t` in `src/driver/mpi_communicator.cpp`
- CMake option `MIST_USE_MPI` for conditional compilation
- Tested with `mpirun -n 4 ./advect1d`

**Build with MPI:**
```bash
cmake . -DMIST_USE_MPI=ON
make
mpirun -n 4 ./examples/advect1d/advect1d
```

**Next steps:**
- Integrate communicator with pipeline for halo exchange
- Add grid_t and distribution_t abstractions
- Per-rank checkpoint/product output
