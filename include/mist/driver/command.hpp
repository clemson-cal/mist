#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>
#include "../archive.hpp"

namespace mist::driver {

// =============================================================================
// Command structs
// =============================================================================

namespace cmd {

struct advance_by {
    std::string var;
    double delta;
    auto fields() const { return std::make_tuple(field("var", var), field("delta", delta)); }
    auto fields() { return std::make_tuple(field("var", var), field("delta", delta)); }
};

struct advance_to {
    std::string var;
    double target;
    auto fields() const { return std::make_tuple(field("var", var), field("target", target)); }
    auto fields() { return std::make_tuple(field("var", var), field("target", target)); }
};

struct set_output {
    std::string format;
    auto fields() const { return std::make_tuple(field("format", format)); }
    auto fields() { return std::make_tuple(field("format", format)); }
};

struct set_physics {
    std::string key;
    std::string value;
    auto fields() const { return std::make_tuple(field("key", key), field("value", value)); }
    auto fields() { return std::make_tuple(field("key", key), field("value", value)); }
};

struct set_initial {
    std::string key;
    std::string value;
    auto fields() const { return std::make_tuple(field("key", key), field("value", value)); }
    auto fields() { return std::make_tuple(field("key", key), field("value", value)); }
};

struct set_exec {
    std::string key;
    std::string value;
    auto fields() const { return std::make_tuple(field("key", key), field("value", value)); }
    auto fields() { return std::make_tuple(field("key", key), field("value", value)); }
};

struct select_timeseries {
    std::vector<std::string> cols;
    auto fields() const { return std::make_tuple(field("cols", cols)); }
    auto fields() { return std::make_tuple(field("cols", cols)); }
};

struct select_products {
    std::vector<std::string> prods;
    auto fields() const { return std::make_tuple(field("prods", prods)); }
    auto fields() { return std::make_tuple(field("prods", prods)); }
};

struct do_timeseries {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct write_physics {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct write_initial {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct write_driver {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct write_profiler {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct write_timeseries {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct write_checkpoint {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct write_products {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct write_iteration {
    std::optional<std::string> dest;
    auto fields() const { return std::make_tuple(field("dest", dest)); }
    auto fields() { return std::make_tuple(field("dest", dest)); }
};

struct repeat_add; // forward declaration - defined after command_t

struct clear_repeat {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct init {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct reset {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct load {
    std::string filename;
    auto fields() const { return std::make_tuple(field("filename", filename)); }
    auto fields() { return std::make_tuple(field("filename", filename)); }
};

struct show_state {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_all {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_physics {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_initial {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_iteration {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_timeseries {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_products {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_profiler {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_driver {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct show_exec {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct help {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct help_schema {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

struct stop {
    auto fields() const { return std::make_tuple(); }
    auto fields() { return std::make_tuple(); }
};

} // namespace cmd

// =============================================================================
// command_t variant (excludes repeat_add to avoid circularity)
// =============================================================================

using command_t = std::variant<
    cmd::advance_by,
    cmd::advance_to,
    cmd::set_output,
    cmd::set_physics,
    cmd::set_initial,
    cmd::set_exec,
    cmd::select_timeseries,
    cmd::select_products,
    cmd::do_timeseries,
    cmd::write_physics,
    cmd::write_initial,
    cmd::write_driver,
    cmd::write_profiler,
    cmd::write_timeseries,
    cmd::write_checkpoint,
    cmd::write_products,
    cmd::write_iteration,
    cmd::clear_repeat,
    cmd::init,
    cmd::reset,
    cmd::load,
    cmd::show_state,
    cmd::show_all,
    cmd::show_physics,
    cmd::show_initial,
    cmd::show_iteration,
    cmd::show_timeseries,
    cmd::show_products,
    cmd::show_profiler,
    cmd::show_driver,
    cmd::show_exec,
    cmd::help,
    cmd::help_schema,
    cmd::stop
>;

// =============================================================================
// repeat_add - now that command_t is defined
// =============================================================================

namespace cmd {

struct repeat_add {
    double interval;
    std::string unit;
    command_t sub_command;
    auto fields() const {
        return std::make_tuple(
            field("interval", interval),
            field("unit", unit),
            field("sub_command", sub_command)
        );
    }
    auto fields() {
        return std::make_tuple(
            field("interval", interval),
            field("unit", unit),
            field("sub_command", sub_command)
        );
    }
};

} // namespace cmd

// =============================================================================
// is_repeatable - only do_*, write_*, show_* commands can be repeated
// =============================================================================

inline auto is_repeatable(const command_t& cmd) -> bool {
    return std::visit([](const auto& c) -> bool {
        using T = std::decay_t<decltype(c)>;
        return std::is_same_v<T, cmd::do_timeseries>
            || std::is_same_v<T, cmd::write_physics>
            || std::is_same_v<T, cmd::write_initial>
            || std::is_same_v<T, cmd::write_driver>
            || std::is_same_v<T, cmd::write_profiler>
            || std::is_same_v<T, cmd::write_timeseries>
            || std::is_same_v<T, cmd::write_checkpoint>
            || std::is_same_v<T, cmd::write_products>
            || std::is_same_v<T, cmd::write_iteration>
            || std::is_same_v<T, cmd::show_state>
            || std::is_same_v<T, cmd::show_all>
            || std::is_same_v<T, cmd::show_physics>
            || std::is_same_v<T, cmd::show_initial>
            || std::is_same_v<T, cmd::show_iteration>
            || std::is_same_v<T, cmd::show_timeseries>
            || std::is_same_v<T, cmd::show_products>
            || std::is_same_v<T, cmd::show_profiler>
            || std::is_same_v<T, cmd::show_driver>
            || std::is_same_v<T, cmd::show_exec>;
    }, cmd);
}

// =============================================================================
// repeating_command_t - stored in state, includes last_executed
// =============================================================================

struct repeating_command_t {
    double interval;
    std::string unit;
    command_t sub_command;
    std::optional<double> last_executed;

    auto time_until_due(double current) const -> double {
        if (!last_executed) return 0.0;
        return *last_executed + interval - current;
    }

    auto fields() const {
        return std::make_tuple(
            field("interval", interval),
            field("unit", unit),
            field("sub_command", sub_command),
            field("last_executed", last_executed)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("interval", interval),
            field("unit", unit),
            field("sub_command", sub_command),
            field("last_executed", last_executed)
        );
    }
};

} // namespace mist::driver
