#pragma once

#include <tuple>
#include <utility>
#include "mist/ndarray.hpp"
#include "mist/comm.hpp"
#include "mist/pipeline.hpp"

#ifdef MIST_WITH_MPI
#include <mpi.h>
#endif

using namespace mist;

// =============================================================================
// Grid Decomposition Pipeline
// =============================================================================

template<std::size_t Rank>
class decomposed_grid;

template<std::size_t Rank>
class distributed_grid;

template<std::size_t Rank>
class grid {
    index_space_t<Rank> space_;

public:
    explicit grid(const index_space_t<Rank>& space) : space_(space) {}

    auto decompose(const uvec_t<Rank>& layout) -> decomposed_grid<Rank> {
        return decomposed_grid<Rank>(space_, layout);
    }
};

// =============================================================================

template<std::size_t Rank>
class decomposed_grid {
    index_space_t<Rank> space_;
    uvec_t<Rank> layout_;

public:
    decomposed_grid(const index_space_t<Rank>& space, const uvec_t<Rank>& layout)
        : space_(space), layout_(layout) {}

    auto distribute(const comm_t& comm) -> distributed_grid<Rank> {
        return distributed_grid<Rank>(space_, layout_, comm);
    }
};

template<std::size_t Rank>
class distributed_grid {
    index_space_t<Rank> space_;
    uvec_t<Rank> layout_;
    comm_t comm_;
    std::vector<index_space_t<Rank>> local_patches_;

    void compute_local_patches() {
        auto total_patches = product(layout_);
        auto patch_range = subspace(index_space(ivec(0), uvec(total_patches)), comm_.size(), comm_.rank(), 0);

        for (auto pi : patch_range) {
            auto patch_space = subspace(space_, layout_, ndindex(pi[0], layout_));
            local_patches_.push_back(patch_space);
        }
    }

public:
    distributed_grid(const index_space_t<Rank>& space, const uvec_t<Rank>& layout, const comm_t& comm)
        : space_(space), layout_(layout), comm_(comm) {
        compute_local_patches();
    }

    template<typename F>
    auto map(F&& fn) -> std::vector<std::invoke_result_t<F, index_space_t<Rank>>> {
        std::vector<std::invoke_result_t<F, index_space_t<Rank>>> results;
        for (const auto& patch_space : local_patches_) {
            results.push_back(fn(patch_space));
        }
        return results;
    }
};

// =============================================================================
// MPI Initialization
// =============================================================================

class mpi_context {
public:
    mpi_context(int argc, char** argv) {
#ifdef MIST_WITH_MPI
        MPI_Init(&argc, &argv);
#else
        (void)argc;
        (void)argv;
#endif
    }

    ~mpi_context() {
#ifdef MIST_WITH_MPI
        MPI_Finalize();
#endif
    }

    auto get_communicator() -> comm_t {
#ifdef MIST_WITH_MPI
        return comm_t::from_mpi(MPI_COMM_WORLD);
#else
        return comm_t{};
#endif
    }
};

// =============================================================================
// Ghost Region Helpers
// =============================================================================

enum class axis { i, j };
enum class region { lo, hi };

auto ghost(const index_space_t<2>& space, region reg, axis ax, int num_guard) -> index_space_t<2> {
    auto lo = start(space);
    auto hi = upper(space);
    auto sh = shape(space);

    if (ax == axis::i) {
        if (reg == region::lo) {
            return index_space(ivec(lo[0] - num_guard, lo[0]), uvec(num_guard, sh[1]));
        } else {
            return index_space(ivec(hi[0], lo[1]), uvec(num_guard, sh[1]));
        }
    } else {
        if (reg == region::lo) {
            return index_space(ivec(lo[0], lo[1] - num_guard), uvec(sh[0], num_guard));
        } else {
            return index_space(ivec(lo[0], hi[1]), uvec(sh[0], num_guard));
        }
    }
}

// =============================================================================
// Transformation: Fluent Pipeline Builder
// =============================================================================

template<typename PatchType, typename... Stages>
class transformation {
    std::tuple<Stages...> stages;

public:
    transformation(const std::tuple<Stages...>& s = {}) : stages(s) {}

    template<parallel::ExchangeStage S>
    auto exchange(S&& stage) -> transformation<PatchType, Stages..., S> {
        auto new_tuple = std::tuple_cat(stages, std::make_tuple(std::forward<S>(stage)));
        return transformation<PatchType, Stages..., S>(new_tuple);
    }

    template<parallel::ComputeStage S>
    auto compute(S&& stage) -> transformation<PatchType, Stages..., S> {
        auto new_tuple = std::tuple_cat(stages, std::make_tuple(std::forward<S>(stage)));
        return transformation<PatchType, Stages..., S>(new_tuple);
    }

    template<parallel::ReduceStage S>
    auto reduce(S&& stage) -> transformation<PatchType, Stages..., S> {
        auto new_tuple = std::tuple_cat(stages, std::make_tuple(std::forward<S>(stage)));
        return transformation<PatchType, Stages..., S>(new_tuple);
    }

    template<typename Scheduler, typename Profiler>
    auto execute(std::vector<PatchType>& patches, comm_t& comm,
                 Scheduler& sched, Profiler& prof) -> void {
        std::apply([&](auto&&... stg) {
            auto pipe = parallel::pipeline(stg...);
            parallel::execute(pipe, patches, comm, sched, prof);
        }, stages);
    }
};
