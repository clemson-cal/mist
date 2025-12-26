#pragma once

#include <tuple>
#include <utility>
#include "mist/ndarray.hpp"
#include "mist/comm.hpp"
#include "mist/pipeline.hpp"

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
    std::vector<index_space_t<Rank>> local_patches_;

public:
    distributed_grid(const index_space_t<Rank>& space, const uvec_t<Rank>& layout, const comm_t& comm) {
        auto total_patches = product(layout);
        auto patch_range = subspace(index_space(ivec(0), uvec(total_patches)), comm.size(), comm.rank(), 0);

        for (auto pi : patch_range) {
            auto patch_space = subspace(space, layout, ndindex(pi[0], layout));
            local_patches_.push_back(patch_space);
        }
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
