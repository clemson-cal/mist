#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "ascii_reader.hpp"
#include "ascii_writer.hpp"
#include "binary_reader.hpp"
#include "binary_writer.hpp"
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

template<typename P>
concept HasValidate = requires(
    const typename P::config_t& cfg,
    const typename P::state_t& s
) {
    { validate(cfg, s) } -> std::same_as<void>;
};

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
// Driver types
// =============================================================================

namespace driver {

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------

enum class scheduling_policy { nearest, exact };

inline const char* to_string(scheduling_policy p) {
    switch (p) {
        case scheduling_policy::nearest: return "nearest";
        case scheduling_policy::exact: return "exact";
    }
    return "unknown";
}

inline scheduling_policy from_string(std::type_identity<scheduling_policy>, const std::string& str) {
    if (str == "nearest") return scheduling_policy::nearest;
    if (str == "exact") return scheduling_policy::exact;
    throw std::runtime_error("scheduling_policy must be 'nearest' or 'exact'");
}

enum class output_format { ascii, binary };

inline const char* to_string(output_format fmt) {
    switch (fmt) {
        case output_format::ascii: return "ascii";
        case output_format::binary: return "binary";
    }
    return "unknown";
}

inline output_format from_string(std::type_identity<output_format>, const std::string& str) {
    if (str == "ascii") return output_format::ascii;
    if (str == "binary") return output_format::binary;
    throw std::runtime_error("output_format must be 'ascii' or 'binary'");
}

inline const char* extension(output_format fmt) {
    return (fmt == output_format::binary) ? ".bin" : ".dat";
}

// -----------------------------------------------------------------------------
// Scheduled output config and state
// -----------------------------------------------------------------------------

struct scheduled_output_config {
    double interval = 1.0;
    int interval_kind = 0;
    scheduling_policy scheduling = scheduling_policy::nearest;

    auto fields() const {
        return std::make_tuple(
            field("interval", interval),
            field("interval_kind", interval_kind),
            field("scheduling", scheduling)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("interval", interval),
            field("interval_kind", interval_kind),
            field("scheduling", scheduling)
        );
    }
};

struct scheduled_output_state {
    int count = 0;
    double next_time = 0.0;

    auto fields() const {
        return std::make_tuple(
            field("count", count),
            field("next_time", next_time)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("count", count),
            field("next_time", next_time)
        );
    }
};

// -----------------------------------------------------------------------------
// Timeseries
// -----------------------------------------------------------------------------

using timeseries_t = std::vector<std::pair<std::string, std::vector<double>>>;

} // namespace driver

template<typename A>
void serialize(A& ar, const char* name, const driver::timeseries_t& value) {
    ar.begin_group(name);
    for (const auto& [col_name, values] : value) {
        ar.write_array(col_name.c_str(), values);
    }
    ar.end_group();
}

template<typename A>
void deserialize(A& ar, const char* name, driver::timeseries_t& value) {
    ar.begin_group(name);
    value.clear();
    while (!ar.at_group_end()) {
        std::string col_name = ar.peek_identifier();
        std::vector<double> values;
        ar.read_array(col_name.c_str(), values);
        value.push_back({col_name, std::move(values)});
    }
    ar.end_group();
}

// -----------------------------------------------------------------------------
// Driver config and state
// -----------------------------------------------------------------------------

namespace driver {

struct config_t {
    int rk_order = 2;
    double cfl = 0.4;
    double t_final = 1.0;
    int max_iter = -1;
    output_format output_format = output_format::ascii;

    scheduled_output_config message{0.1, 0, scheduling_policy::nearest};
    scheduled_output_config checkpoint{1.0, 0, scheduling_policy::nearest};
    scheduled_output_config products{0.1, 0, scheduling_policy::exact};
    scheduled_output_config timeseries{0.01, 0, scheduling_policy::exact};

    auto fields() const {
        return std::make_tuple(
            field("rk_order", rk_order),
            field("cfl", cfl),
            field("t_final", t_final),
            field("max_iter", max_iter),
            field("output_format", output_format),
            field("message", message),
            field("checkpoint", checkpoint),
            field("products", products),
            field("timeseries", timeseries)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("rk_order", rk_order),
            field("cfl", cfl),
            field("t_final", t_final),
            field("max_iter", max_iter),
            field("output_format", output_format),
            field("message", message),
            field("checkpoint", checkpoint),
            field("products", products),
            field("timeseries", timeseries)
        );
    }
};

struct state_t {
    int iteration = 0;
    scheduled_output_state message_state;
    scheduled_output_state checkpoint_state;
    scheduled_output_state products_state;
    scheduled_output_state timeseries_state;
    timeseries_t timeseries;

    auto fields() const {
        return std::make_tuple(
            field("iteration", iteration),
            field("message_state", message_state),
            field("checkpoint_state", checkpoint_state),
            field("products_state", products_state),
            field("timeseries_state", timeseries_state),
            field("timeseries", timeseries)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("iteration", iteration),
            field("message_state", message_state),
            field("checkpoint_state", checkpoint_state),
            field("products_state", products_state),
            field("timeseries_state", timeseries_state),
            field("timeseries", timeseries)
        );
    }
};

} // namespace driver

// =============================================================================
// Combined types (config, state, program)
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

template<Physics P>
struct state {
    driver::state_t driver;
    typename P::state_t physics;

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

template<Physics P>
struct program {
    config<P> config;
    state<P> state;

    auto fields() const {
        return std::make_tuple(
            field("config", config),
            field("state", state)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("config", config),
            field("state", state)
        );
    }
};

// =============================================================================
// Scheduled output
// =============================================================================

template<typename StateT>
class scheduled_output {
public:
    driver::scheduled_output_config config;
    driver::scheduled_output_state* state;
    std::function<void(const StateT&)> callback;

    void validate() const {
        if (config.scheduling == driver::scheduling_policy::exact && config.interval_kind != 0) {
            throw std::runtime_error("exact scheduling requires interval_kind = 0");
        }
    }

    template<typename RKStepFn>
    void handle_exact_output(double t0, double t1, const StateT& physics_state, RKStepFn&& rk_step) {
        if (config.scheduling == driver::scheduling_policy::exact) {
            if (t0 < state->next_time && t1 >= state->next_time) {
                auto state_exact = rk_step(physics_state, state->next_time - t0);
                state->count++;
                state->next_time += config.interval;
                if (callback) callback(state_exact);
            }
        }
    }

    template<typename GetTimeFn>
    void handle_nearest_output(const StateT& physics_state, GetTimeFn&& get_time) {
        if (config.scheduling == driver::scheduling_policy::nearest) {
            if (get_time(physics_state, config.interval_kind) >= state->next_time) {
                state->count++;
                state->next_time += config.interval;
                if (callback) callback(physics_state);
            }
        }
    }
};

// =============================================================================
// I/O functions
// =============================================================================

inline void write_iteration_message(const std::string& message) {
    std::cout << message << std::endl;
}

template<typename WriterT, Physics P>
void write_checkpoint(WriterT& writer, const program<P>& prog) {
    serialize(writer, "checkpoint", prog);
}

template<typename WriterT, Physics P>
void write_products(WriterT& writer, const typename P::product_t& product) {
    serialize(writer, "products", product);
}

template<typename ReaderT, Physics P>
void read_checkpoint(ReaderT& reader, program<P>& prog) {
    deserialize(reader, "checkpoint", prog);
}

// =============================================================================
// Helpers
// =============================================================================

inline void accumulate_timeseries_sample(
    driver::state_t& driver_state,
    const std::vector<std::pair<std::string, double>>& sample)
{
    for (const auto& [name, value] : sample) {
        auto it = std::find_if(
            driver_state.timeseries.begin(),
            driver_state.timeseries.end(),
            [&name](const auto& col) { return col.first == name; }
        );
        if (it != driver_state.timeseries.end()) {
            it->second.push_back(value);
        } else {
            driver_state.timeseries.push_back({name, {value}});
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
typename P::state_t run(program<P>& prog)
{
    using physics_state_t = typename P::state_t;

    auto& driver_config = prog.config.driver;
    auto& physics_config = prog.config.physics;
    auto& driver_state = prog.state.driver;
    auto& physics_state = prog.state.physics;

    if constexpr (HasValidate<P>) {
        validate(physics_config, physics_state);
    }

    auto rk_step = [&](const physics_state_t& s, double dt) -> physics_state_t {
        switch (driver_config.rk_order) {
            case 1: return rk1_step<P>(physics_config, s, dt);
            case 2: return rk2_step<P>(physics_config, s, dt);
            case 3: return rk3_step<P>(physics_config, s, dt);
            default: throw std::runtime_error("rk_order must be 1, 2, or 3");
        }
    };

    if (driver_state.iteration == 0) {
        driver_state.message_state.next_time = driver_config.message.interval;
        driver_state.checkpoint_state.next_time = driver_config.checkpoint.interval;
        driver_state.products_state.next_time = driver_config.products.interval;
        driver_state.timeseries_state.next_time = driver_config.timeseries.interval;
    }

    double last_message_wall_time = get_wall_time();
    int last_message_iteration = driver_state.iteration;
    auto fmt = driver_config.output_format;

    auto message_output = scheduled_output<physics_state_t>{
        driver_config.message,
        &driver_state.message_state,
        [&](const physics_state_t& s) {
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
        }
    };

    auto checkpoint_output = scheduled_output<physics_state_t>{
        driver_config.checkpoint,
        &driver_state.checkpoint_state,
        [&](const physics_state_t& s) {
            char filename[64];
            std::snprintf(filename, sizeof(filename), "chkpt.%04d.%s",
                driver_state.checkpoint_state.count, driver::extension(fmt) + 1);
            program<P> checkpoint_prog{{driver_config, physics_config}, {driver_state, s}};
            if (fmt == driver::output_format::binary) {
                std::ofstream file(filename, std::ios::binary);
                binary_writer writer(file);
                write_checkpoint<binary_writer, P>(writer, checkpoint_prog);
            } else {
                std::ofstream file(filename);
                ascii_writer writer(file);
                write_checkpoint<ascii_writer, P>(writer, checkpoint_prog);
            }
        }
    };

    auto products_output = scheduled_output<physics_state_t>{
        driver_config.products,
        &driver_state.products_state,
        [&](const physics_state_t& s) {
            char filename[64];
            std::snprintf(filename, sizeof(filename), "prods.%04d.%s",
                driver_state.products_state.count, driver::extension(fmt) + 1);
            if (fmt == driver::output_format::binary) {
                std::ofstream file(filename, std::ios::binary);
                binary_writer writer(file);
                write_products<binary_writer, P>(writer, get_product(physics_config, s));
            } else {
                std::ofstream file(filename);
                ascii_writer writer(file);
                write_products<ascii_writer, P>(writer, get_product(physics_config, s));
            }
        }
    };

    auto timeseries_output = scheduled_output<physics_state_t>{
        driver_config.timeseries,
        &driver_state.timeseries_state,
        [&](const physics_state_t& s) {
            accumulate_timeseries_sample(driver_state, timeseries_sample(physics_config, s));
        }
    };

    std::array<scheduled_output<physics_state_t>, 4> outputs = {{
        message_output,
        checkpoint_output,
        products_output,
        timeseries_output
    }};

    for (auto& output : outputs) {
        output.validate();
    }

    if (driver_state.iteration == 0) {
        for (std::size_t i = 1; i < outputs.size(); ++i) {
            outputs[i].callback(physics_state);
        }
    }

    while (true) {
        double t0 = get_time(physics_state, 0);

        if (t0 >= driver_config.t_final) break;
        if (driver_config.max_iter > 0 && driver_state.iteration >= driver_config.max_iter) break;

        double dt = driver_config.cfl * courant_time(physics_config, physics_state);
        double t1 = t0 + dt;

        for (auto& output : outputs) {
            output.handle_exact_output(t0, t1, physics_state, rk_step);
        }

        physics_state = rk_step(physics_state, dt);
        driver_state.iteration++;

        for (auto& output : outputs) {
            output.handle_nearest_output(physics_state,
                [](const auto& s, int kind) { return get_time(s, kind); });
        }
    }

    return physics_state;
}

template<Physics P>
typename P::state_t run(int argc, const char** argv) {
    if (argc < 2) {
        throw std::runtime_error("usage: " + std::string(argv[0]) +
            " <config.cfg | checkpoint.dat | checkpoint.bin> [key=value ...]");
    }

    std::string filename = argv[1];
    bool is_config = filename.size() >= 4 && filename.substr(filename.size() - 4) == ".cfg";
    bool is_checkpoint = filename.size() >= 4 &&
                         (filename.substr(filename.size() - 4) == ".dat" ||
                          filename.substr(filename.size() - 4) == ".bin");

    if (!is_config && !is_checkpoint) {
        throw std::runtime_error("unrecognized file extension: " + filename +
                                 " (expected .cfg, .dat, or .bin)");
    }

    program<P> prog;

    if (is_config) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("cannot open config file: " + filename);
        ascii_reader reader(file);
        deserialize(reader, "config", prog.config);
        prog.state.physics = initial_state(prog.config.physics);
    } else {
        bool is_binary = filename.substr(filename.size() - 4) == ".bin";
        if (is_binary) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) throw std::runtime_error("cannot open checkpoint file: " + filename);
            binary_reader reader(file);
            read_checkpoint<binary_reader, P>(reader, prog);
        } else {
            std::ifstream file(filename);
            if (!file) throw std::runtime_error("cannot open checkpoint file: " + filename);
            ascii_reader reader(file);
            read_checkpoint<ascii_reader, P>(reader, prog);
        }
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto eq = arg.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("invalid override (expected key=value): " + arg);
        }
        set(prog.config, arg.substr(0, eq), arg.substr(eq + 1));
    }

    return run<P>(prog);
}

} // namespace mist
