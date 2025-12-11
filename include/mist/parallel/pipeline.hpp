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
// - key(): returns this stage's unique identifier
// - messages(send): emits messages via callback (no allocation)
// - receive(msg): processes incoming message, returns context when complete

template<typename A>
concept Message = requires(
    A a,
    const A ca,
    typename A::message_t msg,
    std::function<void(typename A::key_t, typename A::message_t)> send
) {
    typename A::key_t;
    typename A::message_t;
    typename A::context_t;
    requires Hashable<typename A::key_t>;
    { ca.key() } -> std::same_as<typename A::key_t>;
    { a.messages(send) } -> std::same_as<void>;
    { a.receive(msg) } -> std::same_as<std::optional<typename A::context_t>>;
};

// Compute stage: performs expensive work suitable for parallel execution
// - value(): consumes the stage and returns the computed context

template<typename A>
concept Compute = requires(A a) {
    typename A::context_t;
    { std::move(a).value() } -> std::same_as<typename A::context_t>;
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
    static_assert(Compute<Stage>, "Stage must satisfy Compute");
    return apply_compute_stages<Context, Rest...>(Stage(std::move(ctx)).value());
}

// =============================================================================
// Execute pipeline: coordinates Message stage, spawns fused Compute stages
// Operates in-place on the input contexts vector
// =============================================================================

template<typename Context, Message M, Compute... Cs, typename Comm, Scheduler Sched>
void execute(
    pipeline<Context, M, Cs...>,
    std::vector<Context>& contexts,
    Comm& comm,
    Sched& sched
) {
    using key_t = typename M::key_t;

    auto results_queue = blocking_queue<std::pair<key_t, Context>>{};
    auto total = contexts.size();
    auto pending = std::unordered_map<key_t, M>{};

    auto spawn_compute = [&](key_t key, Context ctx) {
        sched.spawn([&results_queue, key, c = std::move(ctx)]() mutable {
            auto result = apply_compute_stages<Context, Cs...>(std::move(c));
            results_queue.send({key, std::move(result)});
        });
    };

    auto try_complete = [&](M& stage) -> bool {
        auto key = stage.key();
        while (auto msg = comm.try_recv(key)) {
            if (auto result = stage.receive(std::move(*msg))) {
                spawn_compute(key, std::move(*result));
                return true;
            }
        }
        return false;
    };

    auto drain_pending = [&]() {
        for (auto it = pending.begin(); it != pending.end();) {
            if (try_complete(it->second)) {
                it = pending.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Process each context: create Message stage, emit messages, try to complete
    for (auto& ctx : contexts) {
        auto stage = M(std::move(ctx));
        auto key = stage.key();

        stage.messages([&](key_t dest, auto msg) {
            comm.send(dest, std::move(msg));
        });

        if (!try_complete(stage)) {
            pending.emplace(key, std::move(stage));
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

} // namespace parallel
} // namespace mist
