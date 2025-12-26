#pragma once

#include "comm.hpp"
#include "profiler.hpp"
#include "scheduler.hpp"

namespace mist {

// =============================================================================
// Execution context: library-provided struct for execution resources
// =============================================================================

struct exec_context_t {
    comm_t* comm = nullptr;
    mutable parallel::scheduler_t scheduler;
    mutable perf::profiler_t profiler;

    auto num_threads() const -> std::size_t { return scheduler.num_threads(); }
    auto mpi_size() const -> int { return comm ? comm->size() : 1; }
    auto mpi_rank() const -> int { return comm ? comm->rank() : 0; }

    void set_num_threads(std::size_t n) { scheduler.set_num_threads(n); }
};

} // namespace mist
