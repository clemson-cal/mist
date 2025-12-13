#pragma once

#include <algorithm>
#include <atomic>
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

// Compute stage: per-peer transform (fully parallel)
template<typename S>
concept ComputeStage = requires { &S::value; };

// Exchange stage: peers request/provide data (barrier required)
template<typename S>
concept ExchangeStage = requires {
    typename S::space_t;
    typename S::buffer_t;
    &S::provides;
    &S::data;
};

// Reduce stage: fold across peers then broadcast result (barrier required)
template<typename S>
concept ReduceStage = requires { typename S::value_type; &S::init; &S::reduce; &S::finalize; };

// Type trait to extract Context type from a stage
namespace detail {
    template<typename R, typename S, typename C>
    C extract_context_from_provides(R(S::*)(const C&) const);

    template<typename R, typename S, typename C>
    C extract_context_from_provides(R(S::*)(C) const);

    template<typename C, typename S>
    C extract_context_from_value(C(S::*)(C) const);

    template<typename V, typename S, typename C>
    C extract_context_from_reduce(V(S::*)(V, const C&) const);
}

template<typename S>
struct stage_context;

template<ExchangeStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_provides(&S::provides));
};

template<ReduceStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_reduce(&S::reduce));
};

template<ComputeStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_value(&S::value));
};

template<typename S>
using stage_context_t = typename stage_context<S>::type;

// =============================================================================
// Pipeline: stores stage instances
// =============================================================================

template<typename... Stages>
struct pipeline_t {
    using context_t = stage_context_t<std::tuple_element_t<0, std::tuple<Stages...>>>;
    std::tuple<Stages...> stages;

    template<std::size_t I>
    const auto& get() const { return std::get<I>(stages); }
};

namespace detail {
    template<typename T>
    struct is_pipeline_impl : std::false_type {};

    template<typename... Stages>
    struct is_pipeline_impl<pipeline_t<Stages...>> : std::true_type {};
}

template<typename T>
concept Pipeline = detail::is_pipeline_impl<T>::value;

template<typename... Stages>
auto pipeline(Stages... stages) -> pipeline_t<Stages...> {
    return {std::make_tuple(std::move(stages)...)};
}

// =============================================================================
// compose: fuse multiple compute stages into one
// =============================================================================

template<typename... Stages>
struct composed_t {
    std::tuple<Stages...> stages;

    template<typename Context>
    auto value(Context ctx) const -> Context {
        return apply_stages(std::move(ctx), std::index_sequence_for<Stages...>{});
    }

private:
    template<typename Context, std::size_t... Is>
    auto apply_stages(Context ctx, std::index_sequence<Is...>) const -> Context {
        ((ctx = std::get<Is>(stages).value(std::move(ctx))), ...);
        return ctx;
    }
};

template<typename... Stages>
auto compose(Stages... stages) -> composed_t<Stages...> {
    return {std::make_tuple(std::move(stages)...)};
}

// =============================================================================
// Execute: barrier-based pipeline execution
// =============================================================================

namespace detail {

// Execute a single Exchange stage across all contexts
template<ExchangeStage Stage, typename Context>
void execute_exchange(
    const Stage& stage,
    std::vector<Context>& contexts
) {
    struct request_t {
        std::size_t requester;
        typename Stage::buffer_t buffer;
        typename Stage::space_t requested_space;
    };

    // Collect all requests
    auto requests = std::vector<request_t>{};
    for (std::size_t i = 0; i < contexts.size(); ++i) {
        stage.need(contexts[i], [&](auto buf) {
            requests.push_back({i, buf, space(buf)});
        });
    }

    // Route requests to providers
    for (auto& req : requests) {
        for (std::size_t j = 0; j < contexts.size(); ++j) {
            if (overlaps(req.requested_space, stage.provides(contexts[j]))) {
                copy_overlapping(req.buffer, stage.data(contexts[j]));
            }
        }
    }
}

// Execute a single Compute stage across all contexts (parallel)
template<ComputeStage Stage, typename Context, Scheduler Sched>
void execute_compute(
    const Stage& stage,
    std::vector<Context>& contexts,
    Sched& sched
) {
    auto queue = blocking_queue<std::pair<std::size_t, Context>>{};
    auto n = contexts.size();

    for (std::size_t i = 0; i < n; ++i) {
        sched.spawn([&queue, &stage, i, c = std::move(contexts[i])]() mutable {
            queue.send({i, stage.value(std::move(c))});
        });
    }

    for (std::size_t i = 0; i < n; ++i) {
        auto [idx, ctx] = queue.recv();
        contexts[idx] = std::move(ctx);
    }
}

// Execute a single Reduce stage: fold then broadcast
template<ReduceStage Stage, typename Context>
void execute_reduce(
    const Stage& stage,
    std::vector<Context>& contexts
) {
    auto acc = Stage::init();
    for (const auto& ctx : contexts) {
        acc = stage.reduce(acc, ctx);
    }
    for (auto& ctx : contexts) {
        stage.finalize(acc, ctx);
    }
}

} // namespace detail

// =============================================================================
// Barrier-based pipeline execution
// =============================================================================

namespace detail {

// Dispatch a stage by type
template<typename Stage, typename Context, Scheduler Sched>
void execute_stage(
    const Stage& stage,
    std::vector<Context>& contexts,
    Sched& sched
) {
    if constexpr (ExchangeStage<Stage>) {
        execute_exchange(stage, contexts);
    } else if constexpr (ReduceStage<Stage>) {
        execute_reduce(stage, contexts);
    } else {
        execute_compute(stage, contexts, sched);
    }
}

} // namespace detail

// Execute pipeline: process each stage in sequence with barriers between
template<typename... Stages, Scheduler Sched>
void execute(
    const pipeline_t<Stages...>& pipe,
    std::vector<stage_context_t<std::tuple_element_t<0, std::tuple<Stages...>>>>& contexts,
    Sched& sched
) {
    // Execute each stage in order
    std::apply([&](const Stages&... stages) {
        (detail::execute_stage(stages, contexts, sched), ...);
    }, pipe.stages);
}

// Convenience overload: execute a single stage without wrapping in pipeline
template<typename Stage, typename Context, Scheduler Sched>
    requires (!Pipeline<Stage>)
void execute(
    const Stage& stage,
    std::vector<Context>& contexts,
    Sched& sched
) {
    detail::execute_stage(stage, contexts, sched);
}

} // namespace parallel
} // namespace mist
