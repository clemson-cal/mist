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
};
inline auto fields(const advance_by& c) { return std::make_tuple(field("var", c.var), field("delta", c.delta)); }
inline auto fields(advance_by& c) { return std::make_tuple(field("var", c.var), field("delta", c.delta)); }

struct advance_to {
    std::string var;
    double target;
};
inline auto fields(const advance_to& c) { return std::make_tuple(field("var", c.var), field("target", c.target)); }
inline auto fields(advance_to& c) { return std::make_tuple(field("var", c.var), field("target", c.target)); }

struct set_output {
    std::string format;
};
inline auto fields(const set_output& c) { return std::make_tuple(field("format", c.format)); }
inline auto fields(set_output& c) { return std::make_tuple(field("format", c.format)); }

struct set_physics {
    std::string key;
    std::string value;
};
inline auto fields(const set_physics& c) { return std::make_tuple(field("key", c.key), field("value", c.value)); }
inline auto fields(set_physics& c) { return std::make_tuple(field("key", c.key), field("value", c.value)); }

struct set_initial {
    std::string key;
    std::string value;
};
inline auto fields(const set_initial& c) { return std::make_tuple(field("key", c.key), field("value", c.value)); }
inline auto fields(set_initial& c) { return std::make_tuple(field("key", c.key), field("value", c.value)); }

struct set_exec {
    std::string key;
    std::string value;
};
inline auto fields(const set_exec& c) { return std::make_tuple(field("key", c.key), field("value", c.value)); }
inline auto fields(set_exec& c) { return std::make_tuple(field("key", c.key), field("value", c.value)); }

struct select_timeseries {
    std::vector<std::string> cols;
};
inline auto fields(const select_timeseries& c) { return std::make_tuple(field("cols", c.cols)); }
inline auto fields(select_timeseries& c) { return std::make_tuple(field("cols", c.cols)); }

struct select_products {
    std::vector<std::string> prods;
};
inline auto fields(const select_products& c) { return std::make_tuple(field("prods", c.prods)); }
inline auto fields(select_products& c) { return std::make_tuple(field("prods", c.prods)); }

struct do_timeseries {};
inline auto fields(const do_timeseries&) { return std::make_tuple(); }
inline auto fields(do_timeseries&) { return std::make_tuple(); }

struct write_physics {
    std::optional<std::string> dest;
};
inline auto fields(const write_physics& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_physics& c) { return std::make_tuple(field("dest", c.dest)); }

struct write_initial {
    std::optional<std::string> dest;
};
inline auto fields(const write_initial& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_initial& c) { return std::make_tuple(field("dest", c.dest)); }

struct write_driver {
    std::optional<std::string> dest;
};
inline auto fields(const write_driver& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_driver& c) { return std::make_tuple(field("dest", c.dest)); }

struct write_profiler {
    std::optional<std::string> dest;
};
inline auto fields(const write_profiler& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_profiler& c) { return std::make_tuple(field("dest", c.dest)); }

struct write_timeseries {
    std::optional<std::string> dest;
};
inline auto fields(const write_timeseries& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_timeseries& c) { return std::make_tuple(field("dest", c.dest)); }

struct write_checkpoint {
    std::optional<std::string> dest;
};
inline auto fields(const write_checkpoint& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_checkpoint& c) { return std::make_tuple(field("dest", c.dest)); }

struct write_products {
    std::optional<std::string> dest;
};
inline auto fields(const write_products& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_products& c) { return std::make_tuple(field("dest", c.dest)); }

struct write_iteration {
    std::optional<std::string> dest;
};
inline auto fields(const write_iteration& c) { return std::make_tuple(field("dest", c.dest)); }
inline auto fields(write_iteration& c) { return std::make_tuple(field("dest", c.dest)); }

struct repeat_add; // forward declaration - defined after command_t

struct clear_repeat {};
inline auto fields(const clear_repeat&) { return std::make_tuple(); }
inline auto fields(clear_repeat&) { return std::make_tuple(); }

struct init {};
inline auto fields(const init&) { return std::make_tuple(); }
inline auto fields(init&) { return std::make_tuple(); }

struct reset {};
inline auto fields(const reset&) { return std::make_tuple(); }
inline auto fields(reset&) { return std::make_tuple(); }

struct load {
    std::string filename;
};
inline auto fields(const load& c) { return std::make_tuple(field("filename", c.filename)); }
inline auto fields(load& c) { return std::make_tuple(field("filename", c.filename)); }

struct show_state {};
inline auto fields(const show_state&) { return std::make_tuple(); }
inline auto fields(show_state&) { return std::make_tuple(); }

struct show_all {};
inline auto fields(const show_all&) { return std::make_tuple(); }
inline auto fields(show_all&) { return std::make_tuple(); }

struct show_physics {};
inline auto fields(const show_physics&) { return std::make_tuple(); }
inline auto fields(show_physics&) { return std::make_tuple(); }

struct show_initial {};
inline auto fields(const show_initial&) { return std::make_tuple(); }
inline auto fields(show_initial&) { return std::make_tuple(); }

struct show_iteration {};
inline auto fields(const show_iteration&) { return std::make_tuple(); }
inline auto fields(show_iteration&) { return std::make_tuple(); }

struct show_timeseries {};
inline auto fields(const show_timeseries&) { return std::make_tuple(); }
inline auto fields(show_timeseries&) { return std::make_tuple(); }

struct show_products {};
inline auto fields(const show_products&) { return std::make_tuple(); }
inline auto fields(show_products&) { return std::make_tuple(); }

struct show_profiler {};
inline auto fields(const show_profiler&) { return std::make_tuple(); }
inline auto fields(show_profiler&) { return std::make_tuple(); }

struct show_driver {};
inline auto fields(const show_driver&) { return std::make_tuple(); }
inline auto fields(show_driver&) { return std::make_tuple(); }

struct show_exec {};
inline auto fields(const show_exec&) { return std::make_tuple(); }
inline auto fields(show_exec&) { return std::make_tuple(); }

struct help {};
inline auto fields(const help&) { return std::make_tuple(); }
inline auto fields(help&) { return std::make_tuple(); }

struct help_schema {};
inline auto fields(const help_schema&) { return std::make_tuple(); }
inline auto fields(help_schema&) { return std::make_tuple(); }

struct stop {};
inline auto fields(const stop&) { return std::make_tuple(); }
inline auto fields(stop&) { return std::make_tuple(); }

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
};

inline auto fields(const repeat_add& c) {
    return std::make_tuple(
        field("interval", c.interval),
        field("unit", c.unit),
        field("sub_command", c.sub_command)
    );
}

inline auto fields(repeat_add& c) {
    return std::make_tuple(
        field("interval", c.interval),
        field("unit", c.unit),
        field("sub_command", c.sub_command)
    );
}

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
};

inline auto fields(const repeating_command_t& c) {
    return std::make_tuple(
        field("interval", c.interval),
        field("unit", c.unit),
        field("sub_command", c.sub_command),
        field("last_executed", c.last_executed)
    );
}

inline auto fields(repeating_command_t& c) {
    return std::make_tuple(
        field("interval", c.interval),
        field("unit", c.unit),
        field("sub_command", c.sub_command),
        field("last_executed", c.last_executed)
    );
}

} // namespace mist::driver
