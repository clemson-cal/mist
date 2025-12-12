#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <tuple>
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
// - provides(ctx): returns the index space this context can provide data from
// - need(ctx, request): declares buffers that need data via request(buffer)
// - fill(ctx, provide): provides owned data via provide(data)
// Buffers carry their own index space; topology routes requests to owners.

template<typename S, typename Context, typename Buffer>
concept Exchange = requires(
    Context& ctx,
    const Context& const_ctx,
    std::function<void(Buffer&)> request,
    std::function<void(Buffer&)> provide
) {
    { S::provides(const_ctx) };  // Returns space_t
    { S::need(ctx, request) } -> std::same_as<void>;
    { S::fill(ctx, provide) } -> std::same_as<void>;
};

// Type trait to detect if a stage satisfies Compute
template<typename S, typename Context>
struct is_compute_stage : std::bool_constant<Compute<S, Context>> {};

// Type trait to detect if a stage has need/fill/provides (Exchange-like)
template<typename S, typename = void>
struct is_exchange_stage : std::false_type {};

template<typename S>
struct is_exchange_stage<S, std::void_t<
    decltype(&S::provides),
    decltype(&S::need),
    decltype(&S::fill)
>> : std::true_type {};

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
//
// Required methods:
// - owners(space, contexts): returns indices of contexts owning data in space
// - copy(dst, src, space): copies data from src to dst for the requested region
// - connected(space_a, space_b): returns true if these spaces are neighbors
//   (i.e., one might request guard data that overlaps the other)
//
template<typename T, typename Context>
concept Topology = requires {
    typename T::buffer_t;
    typename T::space_t;
} && requires(
    const T& topo,
    typename T::space_t space_a,
    typename T::space_t space_b,
    const std::vector<Context>& contexts,
    typename T::buffer_t& dst,
    const typename T::buffer_t& src
) {
    // Returns indices of contexts that own data in the requested space
    { topo.owners(space_a, contexts) } -> std::convertible_to<std::vector<std::size_t>>;
    // Copies overlapping data from source to destination
    { topo.copy(dst, src, space_a) } -> std::same_as<void>;
    // Returns true if two spaces are neighbors (could exchange guards)
    { topo.connected(space_a, space_b) } -> std::convertible_to<bool>;
};

// Simple execute: single Exchange stage followed by Compute stages (barrier-based)
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

// =============================================================================
// Generalized pipeline execution: arbitrary Exchange/Compute stage sequences
// Peers proceed eagerly as soon as their dependencies are satisfied.
// =============================================================================

// Request descriptor for guard zone filling
template<typename Buffer, typename Space>
struct guard_request_t {
    Buffer* buffer = nullptr;
    Space requested_space;
    bool fulfilled = false;
};

// Per-peer state for pipeline execution
template<typename Space>
struct peer_state_t {
    std::size_t stage = 0;                      // Current stage index
    bool requests_collected = false;            // Has need() been called this stage?
    bool guards_filled = false;                 // All guard requests fulfilled?
    bool compute_spawned = false;               // Compute task submitted?
    bool compute_done = false;                  // Compute task completed?
    std::vector<std::size_t> potential_requesters;  // Peers that might request from us
    std::vector<std::size_t> requesters_served;     // Peers we've served this stage
    Space provided_space;                       // Index space we provide (from Exchange::provides)
};

// Generalized execute for arbitrary stage sequences
// Uses vectors for dynamic peer counts (can scale to thousands of peers)
template<
    typename Context,
    typename... Stages,
    typename Topo,
    Scheduler Sched
>
    requires Topology<Topo, Context>
void execute(
    pipeline<Context, Stages...>,
    std::vector<Context>& contexts,
    Topo& topo,
    Sched& sched
) {
    constexpr std::size_t num_stages = sizeof...(Stages);
    using buffer_t = typename Topo::buffer_t;
    using space_t = typename Topo::space_t;

    const std::size_t num_peers = contexts.size();
    if (num_peers == 0) return;

    // Per-peer state
    auto peers = std::vector<peer_state_t<space_t>>(num_peers);

    // Pending guard requests: [peer_idx] -> list of requests
    auto pending_requests = std::vector<std::vector<guard_request_t<buffer_t, space_t>>>(num_peers);

    // Queue for compute completion notifications
    auto compute_done_queue = blocking_queue<std::size_t>{};

    // Stage type flags (computed once)
    auto stage_is_exchange = std::vector<bool>(num_stages);
    {
        std::size_t idx = 0;
        ((stage_is_exchange[idx++] = is_exchange_stage<Stages>::value), ...);
    }

    // Helper to get the first Exchange stage type for computing provided spaces
    // We need this to call Exchange::provides on each context
    using first_exchange_or_void = typename std::conditional<
        (is_exchange_stage<Stages>::value || ...),
        std::tuple_element_t<0, std::tuple<Stages...>>,
        void
    >::type;

    // Compute provided spaces and connectivity (potential requesters)
    // This only makes sense if there's at least one Exchange stage
    if constexpr ((is_exchange_stage<Stages>::value || ...)) {
        // Find the first Exchange stage to use for provides()
        // For now, assume all Exchange stages use the same provides()
        auto get_provides = []<typename S>(const Context& ctx) -> space_t {
            if constexpr (is_exchange_stage<S>::value) {
                return S::provides(ctx);
            } else {
                return space_t{};
            }
        };

        // Collect provided spaces
        for (std::size_t i = 0; i < num_peers; ++i) {
            // Use fold expression to find first Exchange stage and call provides
            bool found = false;
            ((found || !(found = is_exchange_stage<Stages>::value) ||
              (peers[i].provided_space = Stages::provides(contexts[i]), true)), ...);
        }

        // Compute connectivity: who might request from whom
        for (std::size_t i = 0; i < num_peers; ++i) {
            for (std::size_t j = 0; j < num_peers; ++j) {
                if (i != j && topo.connected(peers[i].provided_space, peers[j].provided_space)) {
                    peers[i].potential_requesters.push_back(j);
                }
            }
        }
    }

    std::size_t completed_count = 0;

    // Process stages using fold expression to dispatch by type
    auto process_stage = [&]<typename Stage>(std::size_t stage_idx) {
        if constexpr (is_exchange_stage<Stage>::value) {
            // Exchange stage processing
            for (std::size_t peer_idx = 0; peer_idx < num_peers; ++peer_idx) {
                auto& peer = peers[peer_idx];
                if (peer.stage != stage_idx) continue;

                // Collect requests once
                if (!peer.requests_collected) {
                    Stage::need(contexts[peer_idx], [&](buffer_t& buf) {
                        pending_requests[peer_idx].push_back({&buf, space(buf), false});
                    });
                    peer.requests_collected = true;
                }

                // Try to fill pending requests
                bool all_filled = true;
                for (auto& req : pending_requests[peer_idx]) {
                    if (req.fulfilled) continue;

                    // Find provider among peers at this stage
                    for (std::size_t provider_idx = 0; provider_idx < num_peers; ++provider_idx) {
                        if (peers[provider_idx].stage != stage_idx) continue;

                        // Check if provider owns the requested space
                        auto owners = topo.owners(req.requested_space, contexts);
                        bool is_owner = false;
                        for (auto o : owners) {
                            if (o == provider_idx) {
                                is_owner = true;
                                break;
                            }
                        }
                        if (!is_owner) continue;

                        Stage::fill(contexts[provider_idx], [&](buffer_t& src) {
                            topo.copy(*req.buffer, src, req.requested_space);
                        });
                        req.fulfilled = true;

                        // Track that we served this requester
                        auto& served = peers[provider_idx].requesters_served;
                        if (std::find(served.begin(), served.end(), peer_idx) == served.end()) {
                            served.push_back(peer_idx);
                        }
                        break;
                    }

                    if (!req.fulfilled) {
                        all_filled = false;
                    }
                }

                peer.guards_filled = all_filled;
            }
        } else {
            // Compute stage processing
            for (std::size_t peer_idx = 0; peer_idx < num_peers; ++peer_idx) {
                auto& peer = peers[peer_idx];
                if (peer.stage != stage_idx) continue;
                if (peer.compute_spawned) continue;

                peer.compute_spawned = true;
                auto* ctx_ptr = &contexts[peer_idx];

                sched.spawn([ctx_ptr, peer_idx, &compute_done_queue]() {
                    *ctx_ptr = Stage::value(std::move(*ctx_ptr));
                    compute_done_queue.send(peer_idx);
                });
            }
        }
    };

    // Helper to check if any peer is waiting on compute
    auto any_waiting_on_compute = [&]() {
        for (std::size_t peer_idx = 0; peer_idx < num_peers; ++peer_idx) {
            auto& peer = peers[peer_idx];
            if (peer.stage < num_stages &&
                !stage_is_exchange[peer.stage] &&
                peer.compute_spawned &&
                !peer.compute_done) {
                return true;
            }
        }
        return false;
    };

    // Helper to drain completed computes from queue (non-blocking)
    auto drain_compute_done = [&]() {
        while (auto maybe_idx = compute_done_queue.try_recv()) {
            peers[*maybe_idx].compute_done = true;
        }
    };

    // Main loop
    while (completed_count < num_peers) {
        // Drain any completed compute notifications
        drain_compute_done();

        // Process each stage
        std::size_t stage_idx = 0;
        ((process_stage.template operator()<Stages>(stage_idx++)), ...);

        // Advance peers that are ready
        bool any_advanced = false;
        for (std::size_t peer_idx = 0; peer_idx < num_peers; ++peer_idx) {
            auto& peer = peers[peer_idx];
            if (peer.stage >= num_stages) continue;

            bool can_advance = false;

            if (stage_is_exchange[peer.stage]) {
                // Exchange: advance when guards filled AND all potential requesters served
                if (peer.guards_filled) {
                    can_advance = true;
                    for (auto requester : peer.potential_requesters) {
                        if (peers[requester].stage == peer.stage) {
                            // Requester is at same stage - check if we've served them
                            auto& served = peer.requesters_served;
                            if (std::find(served.begin(), served.end(), requester) == served.end()) {
                                can_advance = false;
                                break;
                            }
                        }
                        // If requester is at a later stage, they've already been served
                        // If requester is at an earlier stage, they haven't arrived yet
                        // and we must wait for them
                        if (peers[requester].stage < peer.stage) {
                            can_advance = false;
                            break;
                        }
                    }
                }
            } else {
                // Compute: advance when done
                can_advance = peer.compute_done;
            }

            if (can_advance) {
                peer.stage++;
                // Reset per-stage state for next stage
                peer.requests_collected = false;
                peer.guards_filled = false;
                peer.compute_spawned = false;
                peer.compute_done = false;
                peer.requesters_served.clear();
                pending_requests[peer_idx].clear();

                if (peer.stage >= num_stages) {
                    completed_count++;
                }
                any_advanced = true;
            }
        }

        // If no progress and waiting on compute, block for a completion
        if (!any_advanced && any_waiting_on_compute()) {
            auto done_idx = compute_done_queue.recv();
            peers[done_idx].compute_done = true;
        }
    }
}

} // namespace parallel
} // namespace mist
