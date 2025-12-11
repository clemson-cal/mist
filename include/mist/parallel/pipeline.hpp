#pragma once

#include <concepts>
#include <functional>
#include <vector>
#include "queue.hpp"
#include "scheduler.hpp"

namespace mist {
namespace parallel {

// =============================================================================
// Stage concepts
// =============================================================================

// Compute stage: performs expensive work suitable for parallel execution
// - value(ctx): transforms and returns the context

template<typename S, typename Context>
concept Compute = requires(Context ctx) {
    { S::value(std::move(ctx)) } -> std::same_as<Context>;
};

// Exchange stage: request/provide pattern for guard zone filling
// - need(ctx, request): declares buffers that need data via request(buffer)
// - fill(ctx, provide): provides owned data via provide(data)
// Buffers carry their own index space; topology routes requests to owners.

template<typename S, typename Context, typename Buffer>
concept Exchange = requires(
    Context& ctx,
    std::function<void(Buffer&)> request,
    std::function<void(Buffer&)> provide
) {
    { S::need(ctx, request) } -> std::same_as<void>;
    { S::fill(ctx, provide) } -> std::same_as<void>;
};

// =============================================================================
// Pipeline: type list for composing stages
// =============================================================================

template<typename Context, typename... Stages>
struct pipeline {};

// =============================================================================
// Apply a sequence of Compute stages to a context (fused execution)
// =============================================================================

template<typename Context>
auto apply_compute_stages(Context ctx) -> Context {
    return ctx;
}

template<typename Context, typename Stage, typename... Rest>
auto apply_compute_stages(Context ctx) -> Context {
    static_assert(Compute<Stage, Context>, "Stage must satisfy Compute<Stage, Context>");
    return apply_compute_stages<Context, Rest...>(Stage::value(std::move(ctx)));
}

// =============================================================================
// Execute pipeline with Exchange stage: topology-based guard zone filling
// =============================================================================

// Topology concept: routes requests to owning contexts and copies data
// The topology defines buffer_t and space_t types.
template<typename T, typename Context>
concept Topology = requires {
    typename T::buffer_t;
    typename T::space_t;
} && requires(
    const T& topo,
    typename T::space_t requested_space,
    const std::vector<Context>& contexts,
    typename T::buffer_t& dst,
    const typename T::buffer_t& src
) {
    // Returns indices of contexts that own data in the requested space
    { topo.owners(requested_space, contexts) } -> std::convertible_to<std::vector<std::size_t>>;
    // Copies overlapping data from source to destination
    { topo.copy(dst, src, requested_space) } -> std::same_as<void>;
};

template<
    typename Context,
    typename E,
    typename... Cs,
    typename Topo,
    Scheduler Sched
>
    requires Topology<Topo, Context> && (Compute<Cs, Context> && ...)
void execute(
    pipeline<Context, E, Cs...>,
    std::vector<Context>& contexts,
    Topo& topo,
    Sched& sched
) {
    using buffer_t = typename Topo::buffer_t;
    using space_t = typename Topo::space_t;

    struct request_t {
        std::size_t requester;
        buffer_t* buffer;
        space_t requested_space;
    };

    // Phase 1: Collect all requests
    auto requests = std::vector<request_t>{};
    for (std::size_t i = 0; i < contexts.size(); ++i) {
        E::need(contexts[i], [&](buffer_t& buf) {
            requests.push_back({i, &buf, space(buf)});
        });
    }

    // Phase 2: Route requests to owners and fill
    for (auto& req : requests) {
        auto owner_indices = topo.owners(req.requested_space, contexts);
        for (auto owner_idx : owner_indices) {
            E::fill(contexts[owner_idx], [&](buffer_t& src) {
                topo.copy(*req.buffer, src, req.requested_space);
            });
        }
    }

    // Phase 3: Execute compute stages (can be parallelized)
    auto results_queue = blocking_queue<std::pair<std::size_t, Context>>{};
    auto total = contexts.size();

    for (std::size_t i = 0; i < contexts.size(); ++i) {
        sched.spawn([&results_queue, i, c = std::move(contexts[i])]() mutable {
            auto result = apply_compute_stages<Context, Cs...>(std::move(c));
            results_queue.send({i, std::move(result)});
        });
    }

    // Collect results back into contexts vector
    for (std::size_t i = 0; i < total; ++i) {
        auto [idx, ctx] = results_queue.recv();
        contexts[idx] = std::move(ctx);
    }
}

} // namespace parallel
} // namespace mist
