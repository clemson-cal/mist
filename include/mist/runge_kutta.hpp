#pragma once

#include <concepts>
#include <stdexcept>

namespace mist {

// =============================================================================
// Runge-Kutta Concept
// =============================================================================

template<typename P>
concept RungeKutta = requires(
    typename P::state_t s,
    const typename P::exec_context_t& ctx,
    double dt,
    double alpha) {
    { copy(s, s) } -> std::same_as<void>;
    { rk_step(s, s, dt, alpha, ctx) } -> std::same_as<void>;
};

// =============================================================================
// Runge-Kutta Time Integration
// =============================================================================

template<RungeKutta P>
void rk1_step(
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    copy(temp, state);
    rk_step(state, temp, dt, 1.0, ctx);
}

template<RungeKutta P>
void rk2_step(
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    copy(temp, state);
    rk_step(state, temp, dt, 1.0, ctx);
    rk_step(state, temp, dt, 0.5, ctx);
}

template<RungeKutta P>
void rk3_step(
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    copy(temp, state);
    rk_step(state, temp, dt, 1.0, ctx);
    rk_step(state, temp, dt, 0.25, ctx);
    rk_step(state, temp, dt, 2.0 / 3.0, ctx);
}

template<RungeKutta P>
void rk_advance(
    int order,
    typename P::state_t& state,
    typename P::state_t& temp,
    double dt,
    const typename P::exec_context_t& ctx)
{
    switch (order) {
        case 1: rk1_step<P>(state, temp, dt, ctx); break;
        case 2: rk2_step<P>(state, temp, dt, ctx); break;
        case 3: rk3_step<P>(state, temp, dt, ctx); break;
        default:
            throw std::runtime_error("rk_order must be 1, 2, or 3");
    }
}

} // namespace mist
