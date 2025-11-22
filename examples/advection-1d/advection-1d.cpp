#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "mist/core.hpp"
#include "mist/serialize.hpp"
#include "mist/ascii_reader.hpp"
#include "mist/ascii_writer.hpp"
#include "mist/driver.hpp"

using namespace mist;

// =============================================================================
// 1D Linear Advection Physics Module
// =============================================================================

struct advection_1d {

    // Configuration: runtime parameters
    struct config_t {
        unsigned int num_zones = 100;
        double domain_length = 1.0;
        double advection_velocity = 1.0;

        auto fields() const {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("domain_length", domain_length),
                field("advection_velocity", advection_velocity)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("domain_length", domain_length),
                field("advection_velocity", advection_velocity)
            );
        }
    };

    // State: conservative variables + metadata
    struct state_t {
        std::vector<double> conserved;
        double time;
        index_space_t<1> grid;

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

    // Product: derived quantities
    struct product_t {
        std::vector<double> primitive;
        double total_mass;
        double min_value;
        double max_value;

        auto fields() const {
            return std::make_tuple(
                field("primitive", primitive),
                field("total_mass", total_mass),
                field("min_value", min_value),
                field("max_value", max_value)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("primitive", primitive),
                field("total_mass", total_mass),
                field("min_value", min_value),
                field("max_value", max_value)
            );
        }
    };
};

// Initial state: sine wave
auto initial_state(const advection_1d::config_t& cfg) -> advection_1d::state_t {
    auto grid = index_space(ivec(0), uvec(cfg.num_zones));
    std::vector<double> u(cfg.num_zones);
    double dx = cfg.domain_length / cfg.num_zones;

    for (unsigned int i = 0; i < cfg.num_zones; ++i) {
        double x = (i + 0.5) * dx;
        u[i] = std::sin(2.0 * M_PI * x / cfg.domain_length);
    }

    return {u, 0.0, grid};
}

// Forward Euler step
auto euler_step(
    const advection_1d::config_t& cfg,
    const advection_1d::state_t& state,
    double dt
) -> advection_1d::state_t {

    auto new_state = state;
    new_state.time += dt;

    double dx = cfg.domain_length / cfg.num_zones;
    double v = cfg.advection_velocity;

    // First-order upwind scheme
    for (unsigned int i = 0; i < cfg.num_zones; ++i) {
        unsigned int im1 = (i == 0) ? cfg.num_zones - 1 : i - 1;

        if (v > 0) {
            double flux_left = v * state.conserved[im1];
            double flux_right = v * state.conserved[i];
            new_state.conserved[i] = state.conserved[i] - dt / dx * (flux_right - flux_left);
        } else {
            unsigned int ip1 = (i + 1) % cfg.num_zones;
            double flux_left = v * state.conserved[i];
            double flux_right = v * state.conserved[ip1];
            new_state.conserved[i] = state.conserved[i] - dt / dx * (flux_right - flux_left);
        }
    }

    return new_state;
}

// CFL timestep
auto courant_time(
    const advection_1d::config_t& cfg,
    const advection_1d::state_t& state
) -> double {
    double dx = cfg.domain_length / cfg.num_zones;
    double v = std::abs(cfg.advection_velocity);
    return dx / v;
}

// Weighted average of two states
auto average(
    const advection_1d::state_t& s1,
    const advection_1d::state_t& s2,
    double alpha
) -> advection_1d::state_t {

    auto result = s1;

    for (unsigned int i = 0; i < s1.conserved.size(); ++i) {
        result.conserved[i] = (1.0 - alpha) * s1.conserved[i] + alpha * s2.conserved[i];
    }

    result.time = (1.0 - alpha) * s1.time + alpha * s2.time;
    return result;
}

// Compute diagnostics
auto get_product(
    const advection_1d::config_t& cfg,
    const advection_1d::state_t& state
) -> advection_1d::product_t {

    double dx = cfg.domain_length / cfg.num_zones;
    double total_mass = 0.0;
    double min_val = state.conserved[0];
    double max_val = state.conserved[0];

    for (const auto& u : state.conserved) {
        total_mass += u * dx;
        min_val = std::min(min_val, u);
        max_val = std::max(max_val, u);
    }

    return {state.conserved, total_mass, min_val, max_val};
}

// Get time for scheduling
auto get_time(const advection_1d::state_t& state, int kind) -> double {
    if (kind == 0) return state.time;
    throw std::out_of_range("advection_1d only supports time kind=0");
}

// Zone count for performance metrics
auto zone_count(const advection_1d::state_t& state) -> std::size_t {
    return state.conserved.size();
}

// Timeseries samples
auto timeseries_sample(
    const advection_1d::config_t& cfg,
    const advection_1d::state_t& state
) -> std::vector<std::pair<std::string, double>> {

    auto product = get_product(cfg, state);
    return {
        {"time", state.time},
        {"total_mass", product.total_mass},
        {"min_value", product.min_value},
        {"max_value", product.max_value}
    };
}

// =============================================================================
// Main driver
// =============================================================================

int main(int argc, const char** argv) {
    static_assert(Physics<advection_1d>, "advection_1d must satisfy Physics concept");

    std::cout << "=== 1D Linear Advection Demo (Mist Driver) ===\n\n";

    try {
        auto final_state = run<advection_1d>(argc, argv);
        std::cout << "\n=== Simulation Complete ===\n";
        std::cout << "Final time: " << final_state.time << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
