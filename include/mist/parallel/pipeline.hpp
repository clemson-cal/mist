#pragma once

#include <concepts>
#include <functional>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>
#include "communicator.hpp"
#include "queue.hpp"
#include "scheduler.hpp"

namespace mist {
namespace parallel {

// =============================================================================
// Hashable concept for generic key types
// =============================================================================

template<typename K>
concept Hashable = requires(K k) {
    { std::hash<K>{}(k) } -> std::convertible_to<std::size_t>;
};

// =============================================================================
// Stage concepts: Message and Compute (mutually exclusive)
// =============================================================================

// Message stage: coordinates with peers via message passing
// - key(ctx): returns this context's unique identifier
// - messages(ctx, send): emits messages via callback, returns ctx
// - receive(ctx, msg): processes incoming message, returns (ctx, done)

template<typename S, typename Context>
concept Message = requires(
    Context ctx,
    typename S::message_t&& msg,
    std::function<void(typename S::key_t, typename S::message_t)> send
) {
    typename S::key_t;
    typename S::message_t;
    requires Hashable<typename S::key_t>;
    { S::key(ctx) } -> std::same_as<typename S::key_t>;
    { S::messages(std::move(ctx), send) } -> std::same_as<Context>;
    { S::receive(std::move(ctx), std::move(msg)) } -> std::same_as<std::pair<Context, bool>>;
};

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
// Execute pipeline: coordinates Message stage, spawns fused Compute stages
// Operates in-place on the input contexts vector
// =============================================================================

template<typename Context, typename M, typename... Cs, typename Comm, Scheduler Sched>
    requires Message<M, Context> && (Compute<Cs, Context> && ...)
void execute(
    pipeline<Context, M, Cs...>,
    std::vector<Context>& contexts,
    Comm& comm,
    Sched& sched
) {
    using key_t = typename M::key_t;

    auto results_queue = blocking_queue<std::pair<key_t, Context>>{};
    auto total = contexts.size();
    auto pending = std::unordered_map<key_t, Context>{};

    auto spawn_compute = [&](key_t key, Context ctx) {
        sched.spawn([&results_queue, key, c = std::move(ctx)]() mutable {
            auto result = apply_compute_stages<Context, Cs...>(std::move(c));
            results_queue.send({key, std::move(result)});
        });
    };

    auto try_complete = [&](key_t key, Context ctx) -> std::optional<Context> {
        while (auto msg = comm.try_recv(key)) {
            auto [new_ctx, done] = M::receive(std::move(ctx), std::move(*msg));
            ctx = std::move(new_ctx);
            if (done) {
                spawn_compute(key, std::move(ctx));
                return std::nullopt;
            }
        }
        return ctx;
    };

    auto drain_pending = [&]() {
        for (auto it = pending.begin(); it != pending.end();) {
            auto key = it->first;
            if (auto remaining = try_complete(key, std::move(it->second))) {
                it->second = std::move(*remaining);
                ++it;
            } else {
                it = pending.erase(it);
            }
        }
    };

    // Process each context: emit messages, try to complete
    for (auto& ctx : contexts) {
        auto key = M::key(ctx);

        ctx = M::messages(std::move(ctx), [&](key_t dest, auto msg) {
            comm.send(dest, std::move(msg));
        });

        if (auto remaining = try_complete(key, std::move(ctx))) {
            pending.emplace(key, std::move(*remaining));
        }

        drain_pending();
    }

    // Finish remaining pending stages
    while (!pending.empty()) {
        auto progress = pending.size();
        drain_pending();
        if (pending.size() == progress) {
            std::this_thread::yield();
        }
    }

    // Collect results back into contexts vector
    for (auto i = std::size_t{0}; i < total; ++i) {
        auto [key, ctx] = results_queue.recv();
        contexts[key] = std::move(ctx);
    }
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
