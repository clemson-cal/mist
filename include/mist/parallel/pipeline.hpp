#pragma once

#include <array>
#include <atomic>
#include <bitset>
#include <concepts>
#include <functional>
#include <thread>
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

// Type trait to detect if a stage satisfies Compute
template<typename S, typename Context>
struct is_compute_stage : std::bool_constant<Compute<S, Context>> {};

// Type trait to detect if a stage has need/fill (Exchange-like)
template<typename S, typename = void>
struct is_exchange_stage : std::false_type {};

template<typename S>
struct is_exchange_stage<S, std::void_t<
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

// Extended Topology concept for generalized pipeline execution
// Adds owns() and potential_requesters() for efficient peer-to-peer coordination
template<typename T, typename Context, std::size_t MaxPeers>
concept TopologyExtended = Topology<T, Context> && requires(
    const T& topo,
    std::size_t peer_idx,
    typename T::space_t requested_space
) {
    // Check if peer owns data in the requested space
    { topo.owns(peer_idx, requested_space) } -> std::convertible_to<bool>;
    // Returns bitset of peers that might request data from this peer
    { topo.potential_requesters(peer_idx) } -> std::convertible_to<std::bitset<MaxPeers>>;
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

// Maximum number of guard requests per peer (sufficient for 3D with corners)
constexpr std::size_t max_requests_per_peer = 26;

// Request descriptor for guard zone filling
template<typename Buffer, typename Space>
struct guard_request_t {
    Buffer* buffer = nullptr;
    Space requested_space;
    bool fulfilled = false;
};

// Process a single Exchange stage for peers currently at that stage
template<
    std::size_t MaxPeers,
    std::size_t NumStages,
    typename ExchangeStage,
    typename Context,
    typename Topo
>
void process_exchange_stage(
    std::size_t stage_idx,
    std::array<std::size_t, MaxPeers>& peer_stage,
    std::array<Context, MaxPeers>& peer_ctx,
    std::array<std::bitset<MaxPeers>, NumStages>& guards_filled,
    std::array<std::bitset<MaxPeers>, NumStages>& requests_collected,
    std::array<std::bitset<MaxPeers>, NumStages * MaxPeers>& requesters_served,
    std::array<guard_request_t<typename Topo::buffer_t, typename Topo::space_t>,
               NumStages * MaxPeers * max_requests_per_peer>& pending_requests,
    std::array<std::bitset<max_requests_per_peer>, NumStages * MaxPeers>& request_fulfilled,
    std::size_t num_peers,
    const Topo& topo
) {
    using buffer_t = typename Topo::buffer_t;

    for (std::size_t peer_idx = 0; peer_idx < num_peers; ++peer_idx) {
        if (peer_stage[peer_idx] != stage_idx) {
            continue;
        }

        auto& ctx = peer_ctx[peer_idx];

        // Collect requests once when peer first arrives at this stage
        if (!requests_collected[stage_idx].test(peer_idx)) {
            std::size_t req_idx = 0;
            ExchangeStage::need(ctx, [&](buffer_t& buf) {
                if (req_idx < max_requests_per_peer) {
                    auto idx = stage_idx * MaxPeers * max_requests_per_peer +
                               peer_idx * max_requests_per_peer + req_idx;
                    pending_requests[idx] = {&buf, space(buf), false};
                    req_idx++;
                }
            });
            requests_collected[stage_idx].set(peer_idx);
        }

        // Try to fill pending requests from providers at this stage
        bool all_filled = true;
        for (std::size_t r = 0; r < max_requests_per_peer; ++r) {
            auto req_idx = stage_idx * MaxPeers * max_requests_per_peer +
                           peer_idx * max_requests_per_peer + r;
            auto& req = pending_requests[req_idx];

            if (req.buffer == nullptr) {
                continue;
            }

            auto fulfilled_idx = stage_idx * MaxPeers + peer_idx;
            if (request_fulfilled[fulfilled_idx].test(r)) {
                continue;
            }

            // Find provider among peers at this stage
            for (std::size_t provider_idx = 0; provider_idx < num_peers; ++provider_idx) {
                if (peer_stage[provider_idx] != stage_idx) {
                    continue;
                }
                if (!topo.owns(provider_idx, req.requested_space)) {
                    continue;
                }

                ExchangeStage::fill(peer_ctx[provider_idx], [&](buffer_t& src) {
                    topo.copy(*req.buffer, src, req.requested_space);
                });
                request_fulfilled[fulfilled_idx].set(r);
                requesters_served[stage_idx * MaxPeers + provider_idx].set(peer_idx);
                break;
            }

            if (!request_fulfilled[fulfilled_idx].test(r)) {
                all_filled = false;
            }
        }

        if (all_filled) {
            guards_filled[stage_idx].set(peer_idx);
        }
    }
}

// Process a single Compute stage for peers currently at that stage
template<
    std::size_t MaxPeers,
    std::size_t NumStages,
    typename ComputeStage,
    typename Context,
    Scheduler Sched
>
void process_compute_stage(
    std::size_t stage_idx,
    std::array<std::size_t, MaxPeers>& peer_stage,
    std::array<Context, MaxPeers>& peer_ctx,
    std::array<std::bitset<MaxPeers>, NumStages>& compute_spawned,
    std::array<std::atomic<bool>, NumStages * MaxPeers>& compute_done,
    std::size_t num_peers,
    Sched& sched
) {
    for (std::size_t peer_idx = 0; peer_idx < num_peers; ++peer_idx) {
        if (peer_stage[peer_idx] != stage_idx) {
            continue;
        }
        if (compute_spawned[stage_idx].test(peer_idx)) {
            continue;
        }

        compute_spawned[stage_idx].set(peer_idx);

        auto* ctx_ptr = &peer_ctx[peer_idx];
        auto* done_ptr = &compute_done[stage_idx * MaxPeers + peer_idx];

        sched.spawn([ctx_ptr, done_ptr]() {
            *ctx_ptr = ComputeStage::value(std::move(*ctx_ptr));
            done_ptr->store(true, std::memory_order_release);
        });
    }
}

// Check if a peer can advance from an Exchange stage
template<std::size_t MaxPeers, std::size_t NumStages, typename Topo>
bool can_advance_exchange(
    std::size_t peer_idx,
    std::size_t stage_idx,
    const std::array<std::bitset<MaxPeers>, NumStages>& guards_filled,
    const std::array<std::bitset<MaxPeers>, NumStages * MaxPeers>& requesters_served,
    const Topo& topo
) {
    if (!guards_filled[stage_idx].test(peer_idx)) {
        return false;
    }
    auto potential = topo.potential_requesters(peer_idx);
    auto served = requesters_served[stage_idx * MaxPeers + peer_idx];
    return (potential & ~served).none();
}

// Check if a peer can advance from a Compute stage
template<std::size_t MaxPeers, std::size_t NumStages>
bool can_advance_compute(
    std::size_t peer_idx,
    std::size_t stage_idx,
    const std::array<std::atomic<bool>, NumStages * MaxPeers>& compute_done
) {
    return compute_done[stage_idx * MaxPeers + peer_idx].load(std::memory_order_acquire);
}

// Helper to process stage by index at compile time
template<
    std::size_t StageIdx,
    std::size_t MaxPeers,
    std::size_t NumStages,
    typename Context,
    typename Topo,
    typename Sched,
    typename... Stages
>
struct stage_processor;

// Base case: no more stages
template<
    std::size_t StageIdx,
    std::size_t MaxPeers,
    std::size_t NumStages,
    typename Context,
    typename Topo,
    typename Sched
>
struct stage_processor<StageIdx, MaxPeers, NumStages, Context, Topo, Sched> {
    static void process(
        std::size_t,
        std::array<std::size_t, MaxPeers>&,
        std::array<Context, MaxPeers>&,
        std::array<std::bitset<MaxPeers>, NumStages>&,
        std::array<std::bitset<MaxPeers>, NumStages>&,
        std::array<std::bitset<MaxPeers>, NumStages * MaxPeers>&,
        std::array<guard_request_t<typename Topo::buffer_t, typename Topo::space_t>,
                   NumStages * MaxPeers * max_requests_per_peer>&,
        std::array<std::bitset<max_requests_per_peer>, NumStages * MaxPeers>&,
        std::array<std::bitset<MaxPeers>, NumStages>&,
        std::array<std::atomic<bool>, NumStages * MaxPeers>&,
        std::size_t,
        const Topo&,
        Sched&
    ) {}

    static bool is_exchange(std::size_t) { return false; }
};

// Recursive case
template<
    std::size_t StageIdx,
    std::size_t MaxPeers,
    std::size_t NumStages,
    typename Context,
    typename Topo,
    typename Sched,
    typename Stage,
    typename... Rest
>
struct stage_processor<StageIdx, MaxPeers, NumStages, Context, Topo, Sched, Stage, Rest...> {
    using next = stage_processor<StageIdx + 1, MaxPeers, NumStages, Context, Topo, Sched, Rest...>;

    static void process(
        std::size_t stage_idx,
        std::array<std::size_t, MaxPeers>& peer_stage,
        std::array<Context, MaxPeers>& peer_ctx,
        std::array<std::bitset<MaxPeers>, NumStages>& guards_filled,
        std::array<std::bitset<MaxPeers>, NumStages>& requests_collected,
        std::array<std::bitset<MaxPeers>, NumStages * MaxPeers>& requesters_served,
        std::array<guard_request_t<typename Topo::buffer_t, typename Topo::space_t>,
                   NumStages * MaxPeers * max_requests_per_peer>& pending_requests,
        std::array<std::bitset<max_requests_per_peer>, NumStages * MaxPeers>& request_fulfilled,
        std::array<std::bitset<MaxPeers>, NumStages>& compute_spawned,
        std::array<std::atomic<bool>, NumStages * MaxPeers>& compute_done,
        std::size_t num_peers,
        const Topo& topo,
        Sched& sched
    ) {
        if (stage_idx == StageIdx) {
            if constexpr (is_exchange_stage<Stage>::value) {
                process_exchange_stage<MaxPeers, NumStages, Stage>(
                    stage_idx, peer_stage, peer_ctx, guards_filled, requests_collected,
                    requesters_served, pending_requests, request_fulfilled, num_peers, topo
                );
            } else {
                process_compute_stage<MaxPeers, NumStages, Stage>(
                    stage_idx, peer_stage, peer_ctx, compute_spawned, compute_done, num_peers, sched
                );
            }
        } else {
            next::process(
                stage_idx, peer_stage, peer_ctx, guards_filled, requests_collected,
                requesters_served, pending_requests, request_fulfilled, compute_spawned,
                compute_done, num_peers, topo, sched
            );
        }
    }

    static bool is_exchange(std::size_t stage_idx) {
        if (stage_idx == StageIdx) {
            return is_exchange_stage<Stage>::value;
        }
        return next::is_exchange(stage_idx);
    }
};

// Generalized execute for arbitrary stage sequences
template<
    std::size_t MaxPeers,
    typename Context,
    typename... Stages,
    typename Topo,
    Scheduler Sched
>
    requires TopologyExtended<Topo, Context, MaxPeers>
void execute(
    pipeline<Context, Stages...>,
    std::array<Context, MaxPeers>& contexts,
    std::size_t num_peers,
    Topo& topo,
    Sched& sched
) {
    constexpr std::size_t num_stages = sizeof...(Stages);
    using buffer_t = typename Topo::buffer_t;
    using space_t = typename Topo::space_t;
    using processor = stage_processor<0, MaxPeers, num_stages, Context, Topo, Sched, Stages...>;

    // Per-peer state
    auto peer_stage = std::array<std::size_t, MaxPeers>{};
    for (std::size_t i = 0; i < num_peers; ++i) {
        peer_stage[i] = 0;
    }
    for (std::size_t i = num_peers; i < MaxPeers; ++i) {
        peer_stage[i] = num_stages; // Mark unused slots as completed
    }

    // Per-stage state for Exchange
    auto guards_filled = std::array<std::bitset<MaxPeers>, num_stages>{};
    auto requests_collected = std::array<std::bitset<MaxPeers>, num_stages>{};
    auto requesters_served = std::array<std::bitset<MaxPeers>, num_stages * MaxPeers>{};
    auto pending_requests = std::array<guard_request_t<buffer_t, space_t>,
                                        num_stages * MaxPeers * max_requests_per_peer>{};
    auto request_fulfilled = std::array<std::bitset<max_requests_per_peer>, num_stages * MaxPeers>{};

    // Per-stage state for Compute
    auto compute_spawned = std::array<std::bitset<MaxPeers>, num_stages>{};
    auto compute_done = std::array<std::atomic<bool>, num_stages * MaxPeers>{};
    for (auto& d : compute_done) {
        d.store(false, std::memory_order_relaxed);
    }

    std::size_t completed_count = 0;

    while (completed_count < num_peers) {
        // Process each stage
        for (std::size_t stage_idx = 0; stage_idx < num_stages; ++stage_idx) {
            processor::process(
                stage_idx, peer_stage, contexts, guards_filled, requests_collected,
                requesters_served, pending_requests, request_fulfilled, compute_spawned,
                compute_done, num_peers, topo, sched
            );
        }

        // Advance peers that are ready
        for (std::size_t peer_idx = 0; peer_idx < num_peers; ++peer_idx) {
            auto stage_idx = peer_stage[peer_idx];
            if (stage_idx >= num_stages) {
                continue;
            }

            bool can_advance = false;
            if (processor::is_exchange(stage_idx)) {
                can_advance = can_advance_exchange<MaxPeers, num_stages>(
                    peer_idx, stage_idx, guards_filled, requesters_served, topo
                );
            } else {
                can_advance = can_advance_compute<MaxPeers, num_stages>(
                    peer_idx, stage_idx, compute_done
                );
            }

            if (can_advance) {
                peer_stage[peer_idx] = stage_idx + 1;
                if (peer_stage[peer_idx] >= num_stages) {
                    completed_count++;
                }
            }
        }

        // Yield to allow compute tasks to make progress
        std::this_thread::yield();
    }
}

} // namespace parallel
} // namespace mist
