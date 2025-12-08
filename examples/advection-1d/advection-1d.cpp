#include <cmath>
#include <fstream>
#include <iostream>
#include "mist/ascii_reader.hpp"
#include "mist/ascii_writer.hpp"
#include "mist/core.hpp"
#include "mist/driver.hpp"
#include "mist/ndarray.hpp"
#include "mist/runge_kutta.hpp"
#include "mist/serialize.hpp"

using namespace mist;

// =============================================================================
// 1D Linear Advection Physics Module
// =============================================================================

struct advection_1d {

    // Configuration: runtime parameters
    struct config_t {
        int rk_order = 2;
        double cfl = 0.4;
        double advection_velocity = 1.0;

        auto fields() const {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("advection_velocity", advection_velocity)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("advection_velocity", advection_velocity)
            );
        }
    };

    // Initial data parameters
    struct initial_t {
        unsigned int num_zones = 200;
        double domain_length = 1.0;

        auto fields() const {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("domain_length", domain_length)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("domain_length", domain_length)
            );
        }
    };

    // State: conservative variables + time only
    struct state_t {
        cached_t<double, 1> conserved;
        double time;

        state_t()
            : conserved(index_space(ivec(0), uvec(1))), time(0.0) {}

        state_t(cached_t<double, 1>&& c, double t)
            : conserved(std::move(c)), time(t) {}

        state_t(state_t&&) = default;
        state_t& operator=(state_t&&) = default;

        auto fields() const {
            return std::make_tuple(
                field("conserved", conserved),
                field("time", time)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("conserved", conserved),
                field("time", time)
            );
        }
    };

    // Product: vector of 1D ndarrays (single partition for non-decomposed solver)
    using product_t = std::vector<cached_t<double, 1>>;

    // Execution context: references to config/initial + resources
    struct exec_context_t {
        const config_t& config;
        const initial_t& initial;
        mutable state_t rk_temp;

        exec_context_t(const config_t& cfg, const initial_t& ini)
            : config(cfg), initial(ini), rk_temp() {}
    };
};

// Default physics configuration
auto default_physics_config(std::type_identity<advection_1d>) -> advection_1d::config_t {
    return {2, 0.4, 1.0};
}

// Default initial configuration
auto default_initial_config(std::type_identity<advection_1d>) -> advection_1d::initial_t {
    return {200, 1.0};
}

// Initial state: sine wave
auto initial_state(const advection_1d::exec_context_t& ctx) -> advection_1d::state_t {
    const auto& ini = ctx.initial;
    const auto space = index_space(ivec(0), uvec(ini.num_zones));
    auto u = cached_t<double, 1>(space);
    const auto dx = ini.domain_length / ini.num_zones;

    for (auto i = 0u; i < ini.num_zones; ++i) {
        const auto x = (i + 0.5) * dx;
        u(ivec(i)) = std::sin(2.0 * M_PI * x / ini.domain_length);
    }

    return {std::move(u), 0.0};
}

// Copy state
void copy(advection_1d::state_t& dest, const advection_1d::state_t& source) {
    mist::copy(dest.conserved, source.conserved);
    dest.time = source.time;
}

// RK step function - called via ADL from runge_kutta.hpp
void rk_step(
    advection_1d::state_t& state,
    const advection_1d::state_t& s_base,
    double dt,
    double alpha,
    const advection_1d::exec_context_t& ctx)
{
    const auto n = size(state.conserved);
    const auto dx = ctx.initial.domain_length / n;
    const auto v = ctx.config.advection_velocity;
    const auto space = index_space(ivec(0), uvec(n));
    auto new_conserved = cached_t<double, 1>(space);

    // First-order upwind scheme with periodic boundaries
    for (auto i = 0u; i < n; ++i) {
        const auto im1 = (i == 0) ? n - 1 : i - 1;

        if (v > 0) {
            const auto flux_left = v * state.conserved(ivec(im1));
            const auto flux_right = v * state.conserved(ivec(i));
            new_conserved(ivec(i)) = state.conserved(ivec(i)) - dt / dx * (flux_right - flux_left);
        } else {
            const auto ip1 = (i + 1) % n;
            const auto flux_left = v * state.conserved(ivec(i));
            const auto flux_right = v * state.conserved(ivec(ip1));
            new_conserved(ivec(i)) = state.conserved(ivec(i)) - dt / dx * (flux_right - flux_left);
        }
    }

    // Blend with base state
    for (auto i = 0u; i < n; ++i) {
        state.conserved(ivec(i)) = (1.0 - alpha) * s_base.conserved(ivec(i)) + alpha * new_conserved(ivec(i));
    }

    state.time = (1.0 - alpha) * s_base.time + alpha * (state.time + dt);
}

// Advance function: implements RK time-stepping
void advance(
    advection_1d::state_t& state,
    double dt,
    const advection_1d::exec_context_t& ctx
) {
    rk_advance<advection_1d>(ctx.config.rk_order, state, ctx.rk_temp, dt, ctx);
}

// CFL timestep
auto courant_time(
    const advection_1d::state_t& state,
    const advection_1d::exec_context_t& ctx
) -> double {
    const auto n = size(state.conserved);
    const auto dx = ctx.initial.domain_length / n;
    const auto v = std::abs(ctx.config.advection_velocity);
    return ctx.config.cfl * dx / v;
}

// Zone count for performance metrics
auto zone_count(const advection_1d::state_t& state, const advection_1d::exec_context_t& ctx) -> std::size_t {
    return ctx.initial.num_zones;
}

// Uniform interface: names of time variables
auto names_of_time(std::type_identity<advection_1d>) -> std::vector<std::string> {
    return {"t"};
}

// Uniform interface: names of timeseries columns
auto names_of_timeseries(std::type_identity<advection_1d>) -> std::vector<std::string> {
    return {"time", "total_mass", "min_value", "max_value"};
}

// Uniform interface: names of products
auto names_of_products(std::type_identity<advection_1d>) -> std::vector<std::string> {
    return {"concentration", "cell_size"};
}

// Uniform interface: get time value by name
auto get_time(
    const advection_1d::state_t& state,
    const std::string& name
) -> double {
    if (name == "t") {
        return state.time;
    }
    throw std::runtime_error("unknown time variable: " + name);
}

// Uniform interface: get timeseries value by name
auto get_timeseries(
    const advection_1d::config_t& cfg,
    const advection_1d::initial_t& ini,
    const advection_1d::state_t& state,
    const std::string& name
) -> double {
    if (name == "time") {
        return state.time;
    }

    const auto n = size(state.conserved);
    const auto dx = ini.domain_length / n;

    if (name == "total_mass") {
        auto total_mass = 0.0;
        for (auto i = 0u; i < n; ++i) {
            total_mass += state.conserved(ivec(i)) * dx;
        }
        return total_mass;
    } else if (name == "min_value") {
        auto min_val = state.conserved(ivec(0));
        for (auto i = 0u; i < n; ++i) {
            min_val = std::min(min_val, state.conserved(ivec(i)));
        }
        return min_val;
    } else if (name == "max_value") {
        auto max_val = state.conserved(ivec(0));
        for (auto i = 0u; i < n; ++i) {
            max_val = std::max(max_val, state.conserved(ivec(i)));
        }
        return max_val;
    }

    throw std::runtime_error("unknown timeseries column: " + name);
}

// Uniform interface: get product by name
auto get_product(
    const advection_1d::state_t& state,
    const std::string& name,
    const advection_1d::exec_context_t& ctx
) -> advection_1d::product_t {
    const auto n = ctx.initial.num_zones;
    const auto dx = ctx.initial.domain_length / n;
    const auto s = index_space(ivec(0), uvec(n));

    auto result = advection_1d::product_t{};

    if (name == "concentration") {
        auto arr = cached_t<double, 1>(s);
        for (auto i = 0u; i < n; ++i) {
            arr(ivec(i)) = state.conserved(ivec(i));
        }
        result.push_back(std::move(arr));
        return result;
    } else if (name == "cell_size") {
        result.push_back(cache(fill<double>(s, dx), memory::host, exec::cpu));
        return result;
    }

    throw std::runtime_error("unknown product: " + name);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
    // Create program with default physics and initial configs
    mist::program_t<advection_1d> prog;
    prog.physics = default_physics_config(std::type_identity<advection_1d>{});
    prog.initial = default_initial_config(std::type_identity<advection_1d>{});
    // Note: physics_state is now optional and must be initialized with 'init' command

    // Run interactive simulation
    auto final_state = mist::run(prog);

    return 0;
}
