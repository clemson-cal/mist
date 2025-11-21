#pragma once

#include <concepts>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <chrono>
#include <algorithm>
#include <array>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "ascii_writer.hpp"
#include "serialize.hpp"

namespace mist {

// =============================================================================
// Physics concept
// =============================================================================

template<typename P>
concept Physics = requires(
    typename P::config_t cfg,
    typename P::state_t s,
    typename P::product_t p,
    double dt,
    double alpha,
    int kind
) {
    typename P::config_t;
    typename P::state_t;
    typename P::product_t;

    { initial_state(cfg) } -> std::same_as<typename P::state_t>;
    { euler_step(cfg, s, dt) } -> std::same_as<typename P::state_t>;
    { courant_time(cfg, s) } -> std::same_as<double>;
    { average(s, s, alpha) } -> std::same_as<typename P::state_t>;
    { get_product(cfg, s) } -> std::same_as<typename P::product_t>;
    { get_time(s, kind) } -> std::same_as<double>;
    { zone_count(s) } -> std::same_as<std::size_t>;
    { timeseries_sample(cfg, s) } -> std::same_as<std::vector<std::pair<std::string, double>>>;
} && HasConstFields<typename P::state_t>
  && HasConstFields<typename P::product_t>;

// =============================================================================
// Time integrators
// =============================================================================

template<Physics P>
typename P::state_t rk1_step(
    const typename P::config_t& cfg,
    const typename P::state_t& s0,
    double dt)
{
    return euler_step(cfg, s0, dt);
}

template<Physics P>
typename P::state_t rk2_step(
    const typename P::config_t& cfg,
    const typename P::state_t& s0,
    double dt)
{
    auto s1 = euler_step(cfg, s0, dt);
    auto s2 = euler_step(cfg, s1, dt);
    return average(s0, s2, 0.5);
}

template<Physics P>
typename P::state_t rk3_step(
    const typename P::config_t& cfg,
    const typename P::state_t& s0,
    double dt)
{
    auto s1 = euler_step(cfg, s0, dt);
    auto s2 = euler_step(cfg, s1, dt);
    auto s3 = euler_step(cfg, average(s0, s2, 0.25), dt);
    return average(s0, s3, 2.0 / 3.0);
}

// =============================================================================
// Scheduling policy
// =============================================================================

enum class scheduling_policy { nearest, exact };

inline scheduling_policy parse_scheduling_policy(const std::string& str) {
    if (str == "exact") return scheduling_policy::exact;
    if (str == "nearest") return scheduling_policy::nearest;
    throw std::runtime_error("scheduling policy must be 'exact' or 'nearest'");
}

// =============================================================================
// Scheduled output abstraction
// =============================================================================

template<typename StateT>
class scheduled_output {
public:
    double interval;
    int interval_kind;
    scheduling_policy policy;
    double* next_time;
    int* count;
    std::function<void(const StateT&)> callback;

    void validate() const {
        if (policy == scheduling_policy::exact && interval_kind != 0) {
            throw std::runtime_error("exact scheduling requires interval_kind = 0");
        }
    }

    template<typename RKStepFn>
    void handle_exact_output(double t0, double t1, const StateT& state, RKStepFn&& rk_step) {
        if (policy == scheduling_policy::exact && interval_kind == 0) {
            if (t0 < *next_time && t1 >= *next_time) {
                auto state_exact = rk_step(state, *next_time - t0);
                (*count)++;
                *next_time += interval;
                if (callback) callback(state_exact);
            }
        }
    }

    template<typename GetTimeFn>
    void handle_nearest_output(const StateT& state, GetTimeFn&& get_time) {
        if (policy == scheduling_policy::nearest) {
            if (get_time(state, interval_kind) >= *next_time) {
                (*count)++;
                *next_time += interval;
                if (callback) callback(state);
            }
        }
    }
};

// =============================================================================
// Driver state
// =============================================================================

struct driver_state_t {
    int iteration = 0;
    int message_count = 0;
    int checkpoint_count = 0;
    int products_count = 0;
    int timeseries_count = 0;

    double next_message_time = 0.0;
    double next_checkpoint_time = 0.0;
    double next_products_time = 0.0;
    double next_timeseries_time = 0.0;

    std::vector<std::pair<std::string, std::vector<double>>> timeseries_data;
};

// =============================================================================
// Driver configuration (physics-agnostic)
// =============================================================================

namespace driver {

struct config_t {
    int rk_order = 2;
    double cfl = 0.4;

    double t_final = 1.0;
    int max_iter = -1;

    double message_interval = 0.1;
    int message_interval_kind = 0;
    std::string message_scheduling = "nearest";

    double checkpoint_interval = 1.0;
    int checkpoint_interval_kind = 0;
    std::string checkpoint_scheduling = "nearest";

    double products_interval = 0.1;
    int products_interval_kind = 0;
    std::string products_scheduling = "exact";

    double timeseries_interval = 0.01;
    int timeseries_interval_kind = 0;
    std::string timeseries_scheduling = "exact";

    auto fields() const {
        return std::make_tuple(
            field("rk_order", rk_order),
            field("cfl", cfl),
            field("t_final", t_final),
            field("max_iter", max_iter),
            field("message_interval", message_interval),
            field("message_interval_kind", message_interval_kind),
            field("message_scheduling", message_scheduling),
            field("checkpoint_interval", checkpoint_interval),
            field("checkpoint_interval_kind", checkpoint_interval_kind),
            field("checkpoint_scheduling", checkpoint_scheduling),
            field("products_interval", products_interval),
            field("products_interval_kind", products_interval_kind),
            field("products_scheduling", products_scheduling),
            field("timeseries_interval", timeseries_interval),
            field("timeseries_interval_kind", timeseries_interval_kind),
            field("timeseries_scheduling", timeseries_scheduling)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("rk_order", rk_order),
            field("cfl", cfl),
            field("t_final", t_final),
            field("max_iter", max_iter),
            field("message_interval", message_interval),
            field("message_interval_kind", message_interval_kind),
            field("message_scheduling", message_scheduling),
            field("checkpoint_interval", checkpoint_interval),
            field("checkpoint_interval_kind", checkpoint_interval_kind),
            field("checkpoint_scheduling", checkpoint_scheduling),
            field("products_interval", products_interval),
            field("products_interval_kind", products_interval_kind),
            field("products_scheduling", products_scheduling),
            field("timeseries_interval", timeseries_interval),
            field("timeseries_interval_kind", timeseries_interval_kind),
            field("timeseries_scheduling", timeseries_scheduling)
        );
    }
};

} // namespace driver

// =============================================================================
// Combined configuration (driver + physics)
// =============================================================================

template<Physics P>
struct config {
    driver::config_t driver;
    typename P::config_t physics;

    auto fields() const {
        return std::make_tuple(
            field("driver", driver),
            field("physics", physics)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("driver", driver),
            field("physics", physics)
        );
    }
};

// =============================================================================
// Output functions
// =============================================================================

inline void write_iteration_message(const std::string& message) {
    std::cout << message << std::endl;
}

template<Physics P>
void write_checkpoint(int output_num, const typename P::state_t& state, const driver_state_t& driver_state) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "chkpt.%04d.dat", output_num);
    std::ofstream file(filename);
    ascii_writer writer(file);

    writer.begin_group("checkpoint");

    writer.begin_group("driver_state");
    writer.write_scalar("iteration", driver_state.iteration);
    writer.write_scalar("message_count", driver_state.message_count);
    writer.write_scalar("checkpoint_count", driver_state.checkpoint_count);
    writer.write_scalar("products_count", driver_state.products_count);
    writer.write_scalar("timeseries_count", driver_state.timeseries_count);
    writer.write_scalar("next_message_time", driver_state.next_message_time);
    writer.write_scalar("next_checkpoint_time", driver_state.next_checkpoint_time);
    writer.write_scalar("next_products_time", driver_state.next_products_time);
    writer.write_scalar("next_timeseries_time", driver_state.next_timeseries_time);
    writer.end_group();

    serialize(writer, "state", state);

    // Write timeseries data
    writer.begin_group("timeseries");
    for (const auto& [name, values] : driver_state.timeseries_data) {
        writer.write_array(name.c_str(), values);
    }
    writer.end_group();

    writer.end_group();
}

template<Physics P>
void write_products(int output_num, const typename P::state_t& state, const typename P::product_t& product) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "prods.%04d.dat", output_num);
    std::ofstream file(filename);
    ascii_writer writer(file);

    serialize(writer, "products", product);
}



// =============================================================================
// Helpers
// =============================================================================

inline void accumulate_timeseries_sample(
    driver_state_t& driver_state,
    const std::vector<std::pair<std::string, double>>& sample)
{
    for (const auto& [name, value] : sample) {
        auto it = std::find_if(
            driver_state.timeseries_data.begin(),
            driver_state.timeseries_data.end(),
            [&name](const auto& col) { return col.first == name; }
        );

        if (it != driver_state.timeseries_data.end()) {
            it->second.push_back(value);
        } else {
            driver_state.timeseries_data.push_back({name, {value}});
        }
    }
}

inline double get_wall_time() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
// Main driver
// =============================================================================

template<Physics P>
typename P::state_t run(const config<P>& cfg, driver_state_t& driver_state) {
    using state_t = typename P::state_t;
    const auto& drv = cfg.driver;
    const auto& phys = cfg.physics;

    auto rk_step = [&](const state_t& s, double dt) -> state_t {
        switch (drv.rk_order) {
            case 1: return rk1_step<P>(phys, s, dt);
            case 2: return rk2_step<P>(phys, s, dt);
            case 3: return rk3_step<P>(phys, s, dt);
            default: throw std::runtime_error("rk_order must be 1, 2, or 3");
        }
    };

    auto state = initial_state(phys);

    // Initialize scheduling on first run
    if (driver_state.iteration == 0) {
        driver_state.next_message_time = drv.message_interval;
        driver_state.next_checkpoint_time = drv.checkpoint_interval;
        driver_state.next_products_time = drv.products_interval;
        driver_state.next_timeseries_time = drv.timeseries_interval;
    }

    // Session state
    double last_message_wall_time = get_wall_time();
    int last_message_iteration = driver_state.iteration;

    // Iteration message output
    auto message_output = scheduled_output<state_t>(
        drv.message_interval,
        drv.message_interval_kind,
        parse_scheduling_policy(drv.message_scheduling),
        &driver_state.next_message_time,
        &driver_state.message_count,
        [&](const state_t& s) {
            double wall_now = get_wall_time();
            double wall_elapsed = wall_now - last_message_wall_time;
            int iter_elapsed = driver_state.iteration - last_message_iteration;
            double mzps = (wall_elapsed > 0) ? (iter_elapsed * zone_count(s)) / (wall_elapsed * 1e6) : 0.0;

            std::ostringstream oss;
            oss << "[" << std::setw(6) << std::setfill('0') << driver_state.iteration << "] ";
            oss << "t=" << std::fixed << std::setprecision(5) << get_time(s, 0) << " (";

            for (int kind = 1; kind <= 10; ++kind) {
                try {
                    double t = get_time(s, kind);
                    if (kind > 1) oss << " ";
                    oss << kind << ":" << std::fixed << std::setprecision(4) << t;
                } catch (const std::out_of_range&) {
                    break;
                }
            }
            oss << ") Mzps=" << std::fixed << std::setprecision(3) << mzps;

            write_iteration_message(oss.str());
            last_message_wall_time = wall_now;
            last_message_iteration = driver_state.iteration;
        });

    // Checkpoint output
    auto checkpoint_output = scheduled_output<state_t>(
        drv.checkpoint_interval,
        drv.checkpoint_interval_kind,
        parse_scheduling_policy(drv.checkpoint_scheduling),
        &driver_state.next_checkpoint_time,
        &driver_state.checkpoint_count,
        [&](const state_t& s) {
            write_checkpoint<P>(driver_state.checkpoint_count, s, driver_state);
        });

    // Product output
    auto products_output = scheduled_output<state_t>(
        drv.products_interval,
        drv.products_interval_kind,
        parse_scheduling_policy(drv.products_scheduling),
        &driver_state.next_products_time,
        &driver_state.products_count,
        [&](const state_t& s) {
            write_products<P>(driver_state.products_count, s, get_product(phys, s));
        });

    // Timeseries output
    auto timeseries_output = scheduled_output<state_t>(
        drv.timeseries_interval,
        drv.timeseries_interval_kind,
        parse_scheduling_policy(drv.timeseries_scheduling),
        &driver_state.next_timeseries_time,
        &driver_state.timeseries_count,
        [&](const state_t& s) {
            accumulate_timeseries_sample(driver_state, timeseries_sample(phys, s));
        });

    // Collect outputs
    std::array<scheduled_output<state_t>, 4> outputs = {{
        message_output,
        checkpoint_output,
        products_output,
        timeseries_output
    }};

    for (auto& output : outputs) {
        output.validate();
    }

    // Initial outputs at t=0
    if (driver_state.iteration == 0) {
        for (std::size_t i = 1; i < outputs.size(); ++i) {
            outputs[i].callback(state);
        }
    }

    // Main loop
    while (true) {
        double t0 = get_time(state, 0);

        if (t0 >= drv.t_final) break;
        if (drv.max_iter > 0 && driver_state.iteration >= drv.max_iter) break;

        double dt = drv.cfl * courant_time(phys, state);
        double t1 = t0 + dt;

        for (auto& output : outputs) {
            output.handle_exact_output(t0, t1, state, rk_step);
        }

        state = rk_step(state, dt);
        driver_state.iteration++;

        for (auto& output : outputs) {
            output.handle_nearest_output(state,
                [](const auto& s, int kind) { return get_time(s, kind); });
        }
    }

    return state;
}

template<Physics P>
typename P::state_t run(const config<P>& cfg) {
    driver_state_t driver_state;
    return run(cfg, driver_state);
}

} // namespace mist
