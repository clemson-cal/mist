#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
#include <functional>
#include <tuple>
#include <vector>
#include "comm.hpp"
#include "profiler.hpp"
#include "queue.hpp"
#include "scheduler.hpp"

namespace mist {
namespace parallel {

// =============================================================================
// Stage concepts
// =============================================================================

// Check if a stage has a static name member
template<typename S>
concept HasName = requires { { S::name } -> std::convertible_to<const char*>; };

template<typename S>
auto stage_name(std::size_t index) -> std::string {
    if constexpr (HasName<S>) {
        return S::name;
    } else {
        return "stage_" + std::to_string(index);
    }
}

// Compute stage: per-peer transform (fully parallel)
template<typename S>
concept ComputeStage = requires { &S::value; };

// Exchange stage: peers request/provide data (barrier required)
// provides() returns buffer_t directly (contains both space and data)
template<typename S>
concept ExchangeStage = requires {
    typename S::buffer_t;
    &S::provides;
};

// Reduce stage: fold across peers then broadcast result (barrier required)
// init() and combine() are static; extract() and finalize() are instance methods
template<typename S>
concept ReduceStage = requires {
    typename S::value_type;
    &S::init;
    &S::combine;
    &S::extract;
    &S::finalize;
};

// Type trait to extract Context type from a stage
namespace detail {
    template<typename R, typename S, typename C>
    C extract_context_from_provides(R(S::*)(const C&) const);

    template<typename R, typename S, typename C>
    C extract_context_from_provides(R(S::*)(C) const);

    template<typename C, typename S>
    C extract_context_from_value(C(S::*)(C) const);

    template<typename V, typename S, typename C>
    C extract_context_from_extract(V(S::*)(const C&) const);
}

template<typename S>
struct stage_context;

template<ExchangeStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_provides(&S::provides));
};

template<ReduceStage S>
struct stage_context<S> {
    using type = decltype(detail::extract_context_from_extract(&S::extract));
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
template<ExchangeStage Stage, typename Context, typename Comm>
void execute_exchange(
    const Stage& stage,
    std::vector<Context>& contexts,
    Comm& comm
) {
    using dest_view_t = typename Stage::buffer_t;
    using src_view_t = decltype(stage.provides(std::declval<const Context&>()));

    // Collect publications from all local contexts
    auto publications = std::vector<src_view_t>{};
    for (const auto& ctx : contexts) {
        publications.push_back(stage.provides(ctx));
    }

    // Collect requests from all local contexts
    auto requests = std::vector<dest_view_t>{};
    for (auto& ctx : contexts) {
        stage.need(ctx, [&](auto buf) {
            requests.push_back(buf);
        });
    }

    // Build and execute the exchange plan
    auto plan = comm.template build_plan<src_view_t, dest_view_t>(publications, requests);
    comm.exchange(plan);
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
template<ReduceStage Stage, typename Context, typename Comm>
void execute_reduce(
    const Stage& stage,
    std::vector<Context>& contexts,
    Comm& comm
) {
    // Local fold
    auto acc = Stage::init();
    for (const auto& ctx : contexts) {
        acc = Stage::combine(acc, stage.extract(ctx));
    }

    // Global combine across all ranks
    acc = comm.combine(acc, Stage::combine);

    // Broadcast result to all local contexts
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
template<typename Stage, typename Context, typename Comm, Scheduler Sched>
void execute_stage(
    const Stage& stage,
    std::vector<Context>& contexts,
    Comm& comm,
    Sched& sched
) {
    if constexpr (ExchangeStage<Stage>) {
        execute_exchange(stage, contexts, comm);
    } else if constexpr (ReduceStage<Stage>) {
        execute_reduce(stage, contexts, comm);
    } else {
        execute_compute(stage, contexts, sched);
    }
}

} // namespace detail

// Execute pipeline with profiler and communicator
template<typename... Stages, typename Comm, Scheduler Sched, perf::Profiler Prof>
void execute(
    const pipeline_t<Stages...>& pipe,
    std::vector<stage_context_t<std::tuple_element_t<0, std::tuple<Stages...>>>>& contexts,
    Comm& comm,
    Sched& sched,
    Prof& profiler
) {
    std::size_t stage_index = 0;
    std::apply([&](const Stages&... stages) {
        ((profiler.start(),
          detail::execute_stage(stages, contexts, comm, sched),
          profiler.record(stage_name<Stages>(stage_index++))), ...);
    }, pipe.stages);
}

// Convenience: execute with local-only communicator
template<typename... Stages, Scheduler Sched, perf::Profiler Prof>
void execute(
    const pipeline_t<Stages...>& pipe,
    std::vector<stage_context_t<std::tuple_element_t<0, std::tuple<Stages...>>>>& contexts,
    Sched& sched,
    Prof& profiler
) {
    auto comm = comm_t{};
    execute(pipe, contexts, comm, sched, profiler);
}

// Convenience overload: execute a single stage without wrapping in pipeline
template<typename Stage, typename Context, typename Comm, Scheduler Sched, perf::Profiler Prof>
    requires (!Pipeline<Stage>)
void execute(
    const Stage& stage,
    std::vector<Context>& contexts,
    Comm& comm,
    Sched& sched,
    Prof& profiler
) {
    profiler.start();
    detail::execute_stage(stage, contexts, comm, sched);
    profiler.record(stage_name<Stage>(0));
}

// Convenience: execute single stage with local-only communicator
template<typename Stage, typename Context, Scheduler Sched, perf::Profiler Prof>
    requires (!Pipeline<Stage>)
void execute(
    const Stage& stage,
    std::vector<Context>& contexts,
    Sched& sched,
    Prof& profiler
) {
    auto comm = comm_t{};
    profiler.start();
    detail::execute_stage(stage, contexts, comm, sched);
    profiler.record(stage_name<Stage>(0));
}

} // namespace parallel
} // namespace mist
