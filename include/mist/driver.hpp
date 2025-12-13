#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>
#include <readline/history.h>
#include <readline/readline.h>
#include "ascii_reader.hpp"
#include "ascii_writer.hpp"
#include "binary_reader.hpp"
#include "binary_writer.hpp"
#include "serialize.hpp"

// =============================================================================
// ANSI Color Support
// =============================================================================

namespace mist::color {

namespace ansi {
    inline constexpr const char* reset      = "\033[0m";
    inline constexpr const char* bold       = "\033[1m";
    inline constexpr const char* dim        = "\033[2m";

    // Foreground colors
    inline constexpr const char* black      = "\033[30m";
    inline constexpr const char* red        = "\033[31m";
    inline constexpr const char* green      = "\033[32m";
    inline constexpr const char* yellow     = "\033[33m";
    inline constexpr const char* blue       = "\033[34m";
    inline constexpr const char* magenta    = "\033[35m";
    inline constexpr const char* cyan       = "\033[36m";
    inline constexpr const char* white      = "\033[37m";

    // Bright foreground colors
    inline constexpr const char* bright_black   = "\033[90m";
    inline constexpr const char* bright_red     = "\033[91m";
    inline constexpr const char* bright_green   = "\033[92m";
    inline constexpr const char* bright_yellow  = "\033[93m";
    inline constexpr const char* bright_blue    = "\033[94m";
    inline constexpr const char* bright_magenta = "\033[95m";
    inline constexpr const char* bright_cyan    = "\033[96m";
    inline constexpr const char* bright_white   = "\033[97m";
} // namespace ansi

struct scheme_t {
    const char* reset       = ansi::reset;
    const char* iteration   = ansi::cyan;        // [000001]
    const char* label       = ansi::blue;        // t=, orbit=, Mzps=
    const char* value       = ansi::bright_white;// numeric values
    const char* info        = ansi::green;       // info messages
    const char* warning     = ansi::yellow;      // warnings
    const char* error       = ansi::red;         // errors
    const char* prompt      = ansi::bright_cyan; // > prompt
    const char* key         = ansi::magenta;     // config keys
    const char* selected    = ansi::green;       // [+] selected items
    const char* unselected  = ansi::dim;         // [ ] unselected items
    const char* header      = ansi::bold;        // section headers
};

inline scheme_t enabled_scheme() {
    return scheme_t{};
}

inline scheme_t disabled_scheme() {
    return scheme_t{
        "", "", "", "", "", "", "", "", "", "", "", ""
    };
}

inline bool is_tty(int fd) {
    return isatty(fd) != 0;
}

inline bool is_tty(std::ostream& os) {
    if (&os == &std::cout) return is_tty(STDOUT_FILENO);
    if (&os == &std::cerr) return is_tty(STDERR_FILENO);
    return false;
}

inline scheme_t auto_scheme(std::ostream& os) {
    return is_tty(os) ? enabled_scheme() : disabled_scheme();
}

} // namespace mist::color

// =============================================================================
// Signal Handling for Ctrl-C
// =============================================================================

namespace mist::signal {

inline volatile std::sig_atomic_t interrupted = 0;

inline void handler(int) {
    interrupted = 1;
}

// RAII guard to install/restore signal handler
struct interrupt_guard_t {
    std::sig_atomic_t previous_state;
    void (*previous_handler)(int);

    interrupt_guard_t() {
        previous_state = interrupted;
        interrupted = 0;
        previous_handler = std::signal(SIGINT, handler);
    }

    ~interrupt_guard_t() {
        std::signal(SIGINT, previous_handler);
        interrupted = previous_state;
    }

    [[nodiscard]] bool is_interrupted() const {
        return interrupted != 0;
    }

    void clear() {
        interrupted = 0;
    }
};

} // namespace mist::signal

// =============================================================================
// Physics Concept
// =============================================================================

namespace mist {

template<typename P>
concept Physics = requires(
    typename P::config_t cfg,
    typename P::initial_t ini,
    typename P::state_t s,
    typename P::product_t p,
    const typename P::exec_context_t& ctx) {
    typename P::config_t;
    typename P::initial_t;
    typename P::state_t;
    typename P::product_t;
    typename P::exec_context_t;

    { default_physics_config(std::type_identity<P>{}) } -> std::same_as<typename P::config_t>;
    { default_initial_config(std::type_identity<P>{}) } -> std::same_as<typename P::initial_t>;
    { initial_state(ctx) } -> std::same_as<typename P::state_t>;
    { zone_count(s, ctx) } -> std::same_as<std::size_t>;

    // Uniform interface for discovery
    { names_of_time(std::type_identity<P>{}) } -> std::same_as<std::vector<std::string>>;
    { names_of_timeseries(std::type_identity<P>{}) } -> std::same_as<std::vector<std::string>>;
    { names_of_products(std::type_identity<P>{}) } -> std::same_as<std::vector<std::string>>;

    // Uniform interface for access
    { get_time(s, std::string{}) } -> std::same_as<double>;
    { get_timeseries(cfg, ini, s, std::string{}) } -> std::same_as<double>;
    { get_product(s, std::string{}, ctx) } -> std::same_as<typename P::product_t>;

    // Time-stepping: advance by one CFL timestep
    { advance(s, ctx) } -> std::same_as<void>;
};

// =============================================================================
// Driver state and output format
// =============================================================================

namespace driver {

enum class output_format {
    ascii,
    binary,
    hdf5
};

inline const char* to_string(output_format fmt) {
    switch (fmt) {
        case output_format::ascii: return "ascii";
        case output_format::binary: return "binary";
        case output_format::hdf5: return "hdf5";
    }
    return "unknown";
}

inline output_format from_string(std::type_identity<output_format>, const std::string& s) {
    if (s == "ascii") return output_format::ascii;
    if (s == "binary") return output_format::binary;
    if (s == "hdf5") return output_format::hdf5;
    throw std::runtime_error("unknown output format: " + s);
}

[[nodiscard]] inline output_format infer_format_from_filename(std::string_view filename) {
    if (filename.ends_with(".dat") || filename.ends_with(".cfg")) return output_format::ascii;
    if (filename.ends_with(".bin")) return output_format::binary;
    if (filename.ends_with(".h5")) return output_format::hdf5;
    throw std::runtime_error(std::string("cannot infer format from filename extension: ") + std::string(filename));
}

using timeseries_t = std::map<std::string, std::vector<double>>;

struct recurring_command_t {
    double interval = 0.0;
    std::string unit;
    std::string sub_command;
    std::optional<double> last_executed;

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

struct state_t {
    output_format format = output_format::ascii;
    int iteration = 0;
    int checkpoint_count = 0;
    int products_count = 0;
    int timeseries_count = 0;
    timeseries_t timeseries;
    std::vector<recurring_command_t> recurring_commands;
    std::vector<std::string> selected_products;

    auto fields() const {
        return std::make_tuple(
            field("format", format),
            field("iteration", iteration),
            field("checkpoint_count", checkpoint_count),
            field("products_count", products_count),
            field("timeseries_count", timeseries_count),
            field("timeseries", timeseries),
            field("recurring_commands", recurring_commands),
            field("selected_products", selected_products)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("format", format),
            field("iteration", iteration),
            field("checkpoint_count", checkpoint_count),
            field("products_count", products_count),
            field("timeseries_count", timeseries_count),
            field("timeseries", timeseries),
            field("recurring_commands", recurring_commands),
            field("selected_products", selected_products)
        );
    }
};

} // namespace driver

// =============================================================================
// Program type
// =============================================================================

template<Physics P>
struct program_t {
    typename P::config_t physics;
    typename P::initial_t initial;
    std::optional<typename P::state_t> physics_state;
    driver::state_t driver_state;

    auto fields() const {
        return std::make_tuple(
            field("physics", physics),
            field("initial", initial),
            field("physics_state", physics_state),
            field("driver_state", driver_state)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("physics", physics),
            field("initial", initial),
            field("physics_state", physics_state),
            field("driver_state", driver_state)
        );
    }
};

// =============================================================================
// Helpers
// =============================================================================

[[nodiscard]] inline double get_wall_time() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}


// =============================================================================
// Commands
// =============================================================================

struct command_t {
    enum class type {
        increment_n,      // n++, n += X
        increment_var,    // t += X, orbit += X
        target_n,         // n -> X
        target_var,       // t -> X, orbit -> X
        set_output,       // set output=ascii|binary
        set_physics,      // set physics key1=val1 key2=val2 ...
        set_initial,      // set initial key1=val1 key2=val2 ...
        set_exec,         // set exec key1=val1 key2=val2 ...
        select_timeseries,// select timeseries col1 col2 ...
        clear_timeseries, // clear timeseries
        select_products,  // select products [prod1 prod2 ...]
        clear_products,   // clear products
        do_timeseries,    // do timeseries
        write_timeseries, // write timeseries <filename>
        write_checkpoint, // write checkpoint [filename]
        write_products,   // write products [filename]
        repeat_add,       // repeat <interval> <unit> <sub-command>
        repeat_list,      // repeat list
        repeat_clear,     // repeat clear
        init,             // init
        reset,            // reset
        load,             // load <filename>
        show_message,     // show iteration message
        show_all,         // show (no args)
        show_physics,     // show physics
        show_initial,     // show initial
        show_timeseries,  // show timeseries
        show_products,    // show products
        show_driver,      // show driver
        help,             // help
        stop,             // stop|quit|q
        invalid
    };
    type cmd = type::invalid;
    std::optional<int> int_value;
    std::optional<double> double_value;
    std::optional<std::string> string_value;
    std::optional<std::string> var_name;
    std::optional<std::vector<std::string>> string_list;
    std::string error_msg;
};

[[nodiscard]] inline command_t parse_command(std::string_view input) {
    auto iss = std::istringstream{std::string{input}};
    auto first = std::string{};
    iss >> first;

    if (first.empty()) {
        return {command_t::type::invalid, {}, {}, {}, {}, {}, "empty command"};
    }

    auto result = command_t{};

    // n++ or n+=X or n->X
    if (first == "n++" || first == "n") {
        if (first == "n++") {
            result.cmd = command_t::type::increment_n;
            result.int_value = 1;
        } else if (first == "n") {
            auto op = std::string{};
            iss >> op;
            if (op == "+=") {
                result.cmd = command_t::type::increment_n;
                auto val = 0;
                if (!(iss >> val)) {
                    return {command_t::type::invalid, {}, {}, {}, {}, {}, "n += requires integer"};
                }
                result.int_value = val;
            } else if (op == "->") {
                result.cmd = command_t::type::target_n;
                auto val = 0;
                if (!(iss >> val)) {
                    return {command_t::type::invalid, {}, {}, {}, {}, {}, "n -> requires integer"};
                }
                result.int_value = val;
            }
        }
    }
    // VAR += X or VAR -> X (t, orbit, etc.)
    else if (input.find("+=") != std::string_view::npos || input.find("->") != std::string_view::npos) {
        auto var_name = first;
        auto op = std::string{};
        iss >> op;
        auto val = 0.0;
        if (!(iss >> val)) {
            return {command_t::type::invalid, {}, {}, {}, {}, {}, "operator requires numeric value"};
        }
        if (op == "+=") {
            result.cmd = command_t::type::increment_var;
            result.var_name = var_name;
            result.double_value = val;
        } else if (op == "->") {
            result.cmd = command_t::type::target_var;
            result.var_name = var_name;
            result.double_value = val;
        }
    }
    // set output=X | set physics key1=val1 key2=val2 ...
    else if (first == "set") {
        auto what = std::string{};
        iss >> what;

        // Check if this is "set physics key=val ..." or "set initial key=val ..."
        if (what == "physics") {
            result.cmd = command_t::type::set_physics;
            // Collect all key=value pairs as a single string
            auto pairs = std::string{};
            std::getline(iss, pairs);
            // Trim leading whitespace
            if (const auto start = pairs.find_first_not_of(" \t"); start != std::string::npos) {
                pairs = pairs.substr(start);
            }
            if (pairs.empty()) {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "set physics requires key=value pairs"};
            }
            result.string_value = pairs;
        } else if (what == "initial") {
            result.cmd = command_t::type::set_initial;
            // Collect all key=value pairs as a single string
            auto pairs = std::string{};
            std::getline(iss, pairs);
            // Trim leading whitespace
            if (const auto start = pairs.find_first_not_of(" \t"); start != std::string::npos) {
                pairs = pairs.substr(start);
            }
            if (pairs.empty()) {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "set initial requires key=value pairs"};
            }
            result.string_value = pairs;
        } else if (what == "exec") {
            result.cmd = command_t::type::set_exec;
            // Collect all key=value pairs as a single string
            auto pairs = std::string{};
            std::getline(iss, pairs);
            // Trim leading whitespace
            if (const auto start = pairs.find_first_not_of(" \t"); start != std::string::npos) {
                pairs = pairs.substr(start);
            }
            if (pairs.empty()) {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "set exec requires key=value pairs"};
            }
            result.string_value = pairs;
        } else {
            // Parse as "set output=ascii" format
            const auto eq = what.find('=');
            if (eq == std::string::npos) {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "set requires format: set key=value or set physics key1=val1 ..."};
            }
            const auto key = what.substr(0, eq);
            const auto value = what.substr(eq + 1);
            if (key == "output") {
                result.cmd = command_t::type::set_output;
                result.string_value = value;
            } else {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "unknown setting: " + key};
            }
        }
    }
    // select timeseries col1 col2 ... | select products [prod1 prod2 ...]
    else if (first == "select") {
        auto what = std::string{};
        iss >> what;
        if (what == "timeseries") {
            result.cmd = command_t::type::select_timeseries;
            auto columns = std::vector<std::string>{};
            auto col = std::string{};
            while (iss >> col) {
                columns.push_back(col);
            }
            // Empty list is allowed (means all timeseries columns)
            result.string_list = columns;
        } else if (what == "products") {
            result.cmd = command_t::type::select_products;
            auto products = std::vector<std::string>{};
            auto prod = std::string{};
            while (iss >> prod) {
                products.push_back(prod);
            }
            // Empty list is allowed (means all products)
            result.string_list = products;
        } else {
            return {command_t::type::invalid, {}, {}, {}, {}, {}, "unknown: select " + what};
        }
    }
    // clear timeseries | clear products | clear repeat [indexes]
    else if (first == "clear") {
        auto what = std::string{};
        iss >> what;
        if (what == "timeseries") {
            result.cmd = command_t::type::clear_timeseries;
        } else if (what == "products") {
            result.cmd = command_t::type::clear_products;
        } else if (what == "repeat") {
            result.cmd = command_t::type::repeat_clear;
            // Parse optional indices
            auto indices = std::vector<std::string>{};
            auto idx = std::string{};
            while (iss >> idx) {
                indices.push_back(idx);
            }
            result.string_list = indices;
        } else {
            return {command_t::type::invalid, {}, {}, {}, {}, {}, "unknown: clear " + what};
        }
    }
    // do timeseries
    else if (first == "do") {
        auto action = std::string{};
        iss >> action;
        if (action == "timeseries") {
            result.cmd = command_t::type::do_timeseries;
        } else {
            return {command_t::type::invalid, {}, {}, {}, {}, {}, "unknown action: do " + action};
        }
    }
    // write checkpoint|products|timeseries|message [filename|text]
    else if (first == "write") {
        auto what = std::string{};
        iss >> what;
        if (what == "checkpoint") {
            result.cmd = command_t::type::write_checkpoint;
            auto filename = std::string{};
            if (iss >> filename) {
                result.string_value = filename;
            }
        } else if (what == "products") {
            result.cmd = command_t::type::write_products;
            auto filename = std::string{};
            if (iss >> filename) {
                result.string_value = filename;
            }
        } else if (what == "timeseries") {
            result.cmd = command_t::type::write_timeseries;
            auto filename = std::string{};
            if (iss >> filename) {
                result.string_value = filename;
            }
        } else if (what == "message") {
            result.cmd = command_t::type::show_message;
            auto message = std::string{};
            std::getline(iss, message);
            // Trim leading whitespace
            if (const auto start = message.find_first_not_of(" \t"); start != std::string::npos) {
                message = message.substr(start);
            }
            result.string_value = message;
        } else {
            return {command_t::type::invalid, {}, {}, {}, {}, {}, "write requires checkpoint, products, timeseries, or message"};
        }
    }
    // load <filename>
    else if (first == "load") {
        auto filename = std::string{};
        if (!(iss >> filename)) {
            return {command_t::type::invalid, {}, {}, {}, {}, {}, "load requires filename"};
        }
        result.cmd = command_t::type::load;
        result.string_value = filename;
    }
    // repeat <interval> <unit> <sub-command> | repeat list | repeat clear
    else if (first == "repeat") {
        auto action = std::string{};
        iss >> action;
        if (action == "list") {
            result.cmd = command_t::type::repeat_list;
        } else if (action == "clear") {
            result.cmd = command_t::type::repeat_clear;
        } else {
            // Parse as: repeat <interval> <unit> <sub-command>
            // action is actually the interval
            auto interval = 0.0;
            try {
                interval = std::stod(action);
            } catch (...) {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "repeat requires numeric interval"};
            }
            auto unit = std::string{};
            if (!(iss >> unit)) {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "repeat requires time unit"};
            }
            // Rest of line is the sub-command
            auto sub_command = std::string{};
            std::getline(iss, sub_command);
            // Trim leading whitespace
            const auto start = sub_command.find_first_not_of(" \t");
            if (start == std::string::npos || sub_command.empty()) {
                return {command_t::type::invalid, {}, {}, {}, {}, {}, "repeat requires sub-command"};
            }
            sub_command = sub_command.substr(start);

            // Validate sub-command: must start with "do" or "write"
            // This prevents infinite recursion from nested repeat commands
            auto sub_iss = std::istringstream{sub_command};
            auto sub_first = std::string{};
            sub_iss >> sub_first;
            if (sub_first != "do" && sub_first != "write") {
                return {command_t::type::invalid, {}, {}, {}, {}, {},
                        "repeat sub-command must start with 'do' or 'write'"};
            }

            result.cmd = command_t::type::repeat_add;
            result.double_value = interval;
            result.var_name = unit;
            result.string_value = sub_command;
        }
    }
    // init
    else if (first == "init") {
        result.cmd = command_t::type::init;
    }
    // reset
    else if (first == "reset") {
        result.cmd = command_t::type::reset;
    }
    // show [physics|initial|timeseries|products|driver]
    else if (first == "show") {
        auto what = std::string{};
        iss >> what;
        if (what == "all") {
            result.cmd = command_t::type::show_all;
        } else if (what == "physics") {
            result.cmd = command_t::type::show_physics;
        } else if (what == "initial") {
            result.cmd = command_t::type::show_initial;
        } else if (what == "timeseries") {
            result.cmd = command_t::type::show_timeseries;
        } else if (what == "products") {
            result.cmd = command_t::type::show_products;
        } else if (what == "driver") {
            result.cmd = command_t::type::show_driver;
        } else {
            return {command_t::type::invalid, {}, {}, {}, {}, {}, "show requires physics, initial, timeseries, products, driver, or no argument for all"};
        }
    }
    // help
    else if (first == "help") {
        result.cmd = command_t::type::help;
    }
    // stop|quit|q
    else if (first == "stop" || first == "quit" || first == "q") {
        result.cmd = command_t::type::stop;
    }
    else {
        return {command_t::type::invalid, {}, {}, {}, {}, {}, "unknown command: " + first};
    }

    return result;
}

inline const char* help_text = R"(
  ---------------------------------------------------------------------------------------
  Stepping
  ---------------------------------------------------------------------------------------
    n++                            - Advance by 1 iteration
    n += 10                        - Advance by 10 iterations
    n -> 1000                      - Advance to iteration 1000
    t += 10.0                      - Advance time by exactly 10.0
    t -> 20.0                      - Advance time to exactly 20.0
    orbit += 3.0                   - Advance until orbit increases by at least 3.0
    orbit -> 60.0                  - Advance until orbit reaches at least 60.0

  ---------------------------------------------------------------------------------------
  Configuration
  ---------------------------------------------------------------------------------------
    set output=ascii               - Set output format (ascii|binary|hdf5)
    set physics key1=val1 ...      - Set physics config parameters
    set initial key1=val1 ...      - Set initial data parameters (only when state is null)
    set exec key1=val1 ...         - Set execution parameters (e.g. num_threads)
    select products [prod1 ...]    - Select products (no args = all)
    select timeseries [col1 ...]   - Select timeseries columns (no args = all)
    clear timeseries               - Clear timeseries data

  ---------------------------------------------------------------------------------------
  State management
  ---------------------------------------------------------------------------------------
    init                           - Generate initial state from config
    reset                          - Reset driver and clear physics state
    load <file>                    - Load data or command sequence from file
                                     .dat|.bin|.h5 = checkpoint/physics/initial
                                     .prog|.mist   = command sequence

  ---------------------------------------------------------------------------------------
  Sampling
  ---------------------------------------------------------------------------------------
    do timeseries                  - Record timeseries sample

  ---------------------------------------------------------------------------------------
  File I/O
  ---------------------------------------------------------------------------------------
    write timeseries [file]        - Write timeseries to file
    write checkpoint [file]        - Write checkpoint to file
    write products [file]          - Write products to file
    write message <text>           - Write custom message to stdout

  ---------------------------------------------------------------------------------------
  Recurring commands
  ---------------------------------------------------------------------------------------
    repeat <interval> <unit> <cmd> - Execute command every interval ('do' or 'write')
    repeat list                    - List active recurring commands
    repeat clear                   - Clear all recurring commands

  ---------------------------------------------------------------------------------------
  Information
  ---------------------------------------------------------------------------------------
    show all                       - Show all
    show physics                   - Show physics configuration
    show initial                   - Show initial configuration
    show products                  - Show available and selected products
    show timeseries                - Show timeseries data
    show driver                    - Show driver state (including repeating tasks)
    help                           - Show this help
    stop | quit | q                - Exit simulation
)";

inline void print_help() {
    std::cout << "\n" << help_text << std::endl;
}

// =============================================================================
// Driver class - executes commands, no direct I/O
// =============================================================================

template<Physics P>
class driver_t {
    program_t<P>& prog;
    typename P::exec_context_t exec_context;
    std::ostream* out;
    std::ostream* err;
    color::scheme_t colors;
    color::scheme_t err_colors;

    // Performance tracking
    double command_start_wall_time = 0.0;
    int command_start_iteration = 0;
    double last_dt = 0.0;

public:
    explicit driver_t(program_t<P>& prog, std::ostream& out = std::cout, std::ostream& err = std::cerr)
        : prog(prog)
        , exec_context(prog.physics, prog.initial)
        , out(&out)
        , err(&err)
        , colors(color::auto_scheme(out))
        , err_colors(color::auto_scheme(err)) {
        command_start_wall_time = get_wall_time();
        command_start_iteration = prog.driver_state.iteration;
    }

    // Execute a command - returns true to continue, false to stop
    bool execute(const command_t& cmd);

    // Get formatted messages
    [[nodiscard]] std::string iteration_message();

    // // Access to program state
    const program_t<P>& program() const { return prog; }

private:

    // Helper to parse and apply key=value pair
    template<typename T>
    void apply_key_value(T& target, const std::string& pair) {
        const auto eq = pair.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("invalid key=value pair: " + pair);
        }
        const auto key = pair.substr(0, eq);
        const auto value = pair.substr(eq + 1);
        set(target, key, value);
    }

    // Time stepping
    void do_timestep();
    void advance_to_target(const std::string& var_name, double target, std::optional<int> target_iteration = std::nullopt);
    void execute_recurring_commands();

    // Command handlers
    void handle_increment_n(int n);
    void handle_target_n(int target);
    void handle_increment_var(const std::string& kind, double increment);
    void handle_target_var(const std::string& kind, double target);
    void handle_set_output(const std::string& format_str);
    void handle_set_physics(const std::string& pairs_str);
    void handle_set_initial(const std::string& pairs_str);
    void handle_set_exec(const std::string& pairs_str);
    void handle_select_timeseries(std::vector<std::string> columns);
    void handle_clear_timeseries();
    void handle_select_products(std::vector<std::string> products);
    void handle_clear_products();
    void handle_do_timeseries();
    void handle_write_timeseries(const std::optional<std::string>& filename);
    void handle_write_checkpoint(const std::optional<std::string>& filename);
    void handle_write_products(const std::optional<std::string>& filename);
    void handle_repeat_add(double interval, const std::string& unit, const std::string& sub_cmd);
    void handle_repeat_list();
    void handle_repeat_clear(const std::vector<std::string>& indices);
    void handle_init();
    void handle_reset();
    void handle_load(const std::string& filename);
    void handle_show_all();
    void handle_show_message();
    void handle_show_physics();
    void handle_show_initial();
    void handle_show_timeseries();
    void handle_show_products();
    void handle_show_driver();
    void handle_help();
    void handle_stop();
};

// =============================================================================
// Driver implementation
// =============================================================================

template<Physics P>
void driver_t<P>::do_timestep() {
    if (!prog.physics_state.has_value()) {
        throw std::runtime_error("physics state not initialized; use 'init' command first");
    }
    const auto time_names = names_of_time(std::type_identity<P>{});
    const auto t0 = get_time(*prog.physics_state, time_names[0]);
    advance(*prog.physics_state, exec_context);
    const auto t1 = get_time(*prog.physics_state, time_names[0]);
    last_dt = t1 - t0;
    prog.driver_state.iteration++;
}

template<Physics P>
void driver_t<P>::advance_to_target(const std::string& var_name, double target, std::optional<int> target_iteration) {
    if (!prog.physics_state.has_value()) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "physics state not initialized; use 'init' command first\n";
        return;
    }

    auto guard = signal::interrupt_guard_t{};

    // If target_iteration is specified, advance until iteration reaches target
    if (target_iteration.has_value()) {
        while (prog.driver_state.iteration < target_iteration.value() && !guard.is_interrupted()) {
            do_timestep();
            execute_recurring_commands();
        }
        if (guard.is_interrupted()) {
            *err << "\n" << err_colors.warning << "interrupted" << err_colors.reset << "\n";
        }
        return;
    }

    // Advance until time variable reaches or exceeds target
    while (get_time(*prog.physics_state, var_name) < target && !guard.is_interrupted()) {
        do_timestep();
        execute_recurring_commands();
    }

    if (guard.is_interrupted()) {
        *err << "\n" << err_colors.warning << "interrupted" << err_colors.reset << "\n";
    }
}

template<Physics P>
std::string driver_t<P>::iteration_message() {
    if (!prog.physics_state.has_value()) {
        return "[no state initialized]\n";
    }
    const auto wall_now = get_wall_time();
    const auto wall_elapsed = wall_now - command_start_wall_time;
    const auto iter_elapsed = prog.driver_state.iteration - command_start_iteration;
    const auto zps = (wall_elapsed > 0) ? (iter_elapsed * zone_count(*prog.physics_state, exec_context)) / wall_elapsed : 0.0;
    const auto time_names = names_of_time(std::type_identity<P>{});
    const auto& c = colors;

    auto oss = std::ostringstream{};
    oss << c.iteration << "[" << std::setw(6) << std::setfill('0')
        << prog.driver_state.iteration << "]" << c.reset << " ";

    for (std::size_t i = 0; i < time_names.size(); ++i) {
        if (i > 0) oss << " ";
        oss << c.label << time_names[i] << "=" << c.reset
            << c.value << std::scientific << std::showpos << std::setprecision(6)
            << get_time(*prog.physics_state, time_names[i]) << std::noshowpos << c.reset;
    }
    oss << " " << c.label << "dt=" << c.reset
        << c.value << std::scientific << std::setprecision(6) << last_dt << c.reset;
    oss << " " << c.label << "zps=" << c.reset
        << c.value << std::scientific << std::setprecision(2) << zps << c.reset << "\n";
    return oss.str();
}

template<Physics P>
void driver_t<P>::execute_recurring_commands() {
    if (!prog.physics_state.has_value()) {
        return;
    }
    for (auto& rcmd : prog.driver_state.recurring_commands) {
        try {
            const auto current_value = get_time(*prog.physics_state, rcmd.unit);

            // Initialize last_executed on first check
            if (!rcmd.last_executed.has_value()) {
                rcmd.last_executed = current_value;
            }

            if (current_value >= *rcmd.last_executed + rcmd.interval) {
                const auto subcmd = parse_command(rcmd.sub_command);
                if (subcmd.cmd == command_t::type::invalid) {
                    *err << err_colors.error << "error in recurring command: "
                         << err_colors.reset << subcmd.error_msg << "\n";
                    continue;
                }

                // Execute the sub-command (only do/write commands are allowed)
                switch (subcmd.cmd) {
                    case command_t::type::do_timeseries:
                        handle_do_timeseries();
                        break;
                    case command_t::type::write_timeseries:
                        handle_write_timeseries(subcmd.string_value);
                        break;
                    case command_t::type::write_checkpoint:
                        handle_write_checkpoint(subcmd.string_value);
                        break;
                    case command_t::type::write_products:
                        handle_write_products(subcmd.string_value);
                        break;
                    case command_t::type::show_message:
                        handle_show_message();
                        break;
                    default:
                        *err << err_colors.error << "error: " << err_colors.reset
                             << "recurring command type not supported: " << rcmd.sub_command << "\n";
                        break;
                }

                rcmd.last_executed = current_value;
            }
        } catch (const std::exception& e) {
            *err << err_colors.error << "error: recurring command '"
                 << err_colors.reset << e.what() << "'\n";
        }
    }
}

template<Physics P>
void driver_t<P>::handle_increment_n(int n) {
    const auto target = prog.driver_state.iteration + n;
    advance_to_target("", 0.0, target);
    *out << iteration_message();
}

template<Physics P>
void driver_t<P>::handle_target_n(int target) {
    advance_to_target("", 0.0, target);
    *out << iteration_message();
}

template<Physics P>
void driver_t<P>::handle_increment_var(const std::string& kind, double increment) {
    const auto start_value = get_time(*prog.physics_state, kind);
    const auto target = start_value + increment;
    advance_to_target(kind, target);
    *out << iteration_message();
}

template<Physics P>
void driver_t<P>::handle_target_var(const std::string& kind, double target) {
    advance_to_target(kind, target);
    *out << iteration_message();
}

template<Physics P>
void driver_t<P>::handle_set_output(const std::string& format_str) {
    prog.driver_state.format = driver::from_string(std::type_identity<driver::output_format>{}, format_str);
    *out << colors.info << "output format set to " << colors.value << format_str
         << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_set_physics(const std::string& pairs_str) {
    auto pairs_ss = std::istringstream{pairs_str};
    auto pair = std::string{};

    try {
        while (pairs_ss >> pair) {
            apply_key_value(prog.physics, pair);
        }
        *out << colors.info << "physics config updated: " << colors.value << pairs_str
             << colors.reset << "\n";
    } catch (const std::exception& e) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "failed to set physics config: " << e.what() << "\n";
    }
}

template<Physics P>
void driver_t<P>::handle_set_initial(const std::string& pairs_str) {
    // Check if state is initialized - cannot modify initial params if state exists
    if (prog.physics_state.has_value()) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "cannot modify initial params when state is initialized; use 'reset' first\n";
        return;
    }

    auto pairs_ss = std::istringstream{pairs_str};
    auto pair = std::string{};

    try {
        while (pairs_ss >> pair) {
            apply_key_value(prog.initial, pair);
        }
        *out << colors.info << "initial config updated: " << colors.value << pairs_str
             << colors.reset << "\n";
    } catch (const std::exception& e) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "failed to set initial config: " << e.what() << "\n";
    }
}

template<Physics P>
void driver_t<P>::handle_set_exec(const std::string& pairs_str) {
    auto pairs_ss = std::istringstream{pairs_str};
    auto pair = std::string{};

    while (pairs_ss >> pair) {
        const auto eq = pair.find('=');
        if (eq == std::string::npos) {
            *err << err_colors.error << "error: " << err_colors.reset
                 << "invalid key=value pair: " << pair << "\n";
            return;
        }
        const auto key = pair.substr(0, eq);
        const auto value = pair.substr(eq + 1);

        if (key == "num_threads") {
            if constexpr (requires { exec_context.set_num_threads(std::size_t{}); }) {
                exec_context.set_num_threads(std::stoul(value));
                *out << colors.info << "exec num_threads set to " << colors.value << value
                     << colors.reset << "\n";
            } else {
                *err << err_colors.error << "error: " << err_colors.reset
                     << "this physics module does not support num_threads\n";
            }
        } else {
            *err << err_colors.error << "error: " << err_colors.reset
                 << "unknown exec parameter: " << key << "\n";
        }
    }
}

template<Physics P>
void driver_t<P>::handle_select_timeseries(std::vector<std::string> columns) {
    const auto available = names_of_timeseries(std::type_identity<P>{});

    if (columns.empty()) {
        columns = available;
    }
    if (!prog.driver_state.timeseries.empty()) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "use 'clear timeseries' before 'select timeseries ...'\n";
        return;
    }
    for (const auto& col : columns) {
        if (std::find(available.begin(), available.end(), col) == available.end()) {
            *err << err_colors.error << "error: " << err_colors.reset
                 << "column '" << col << "' not found in physics timeseries\n";
            return;
        }
    }
    for (const auto& col : columns) {
        prog.driver_state.timeseries[col] = {};
    }
    handle_show_timeseries();
}

template<Physics P>
void driver_t<P>::handle_clear_timeseries() {
    prog.driver_state.timeseries.clear();
    *out << colors.info << "clear timeseries" << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_select_products(std::vector<std::string> products) {
    const auto available = names_of_products(std::type_identity<P>{});

    if (products.empty()) {
        prog.driver_state.selected_products = available;
        handle_show_products();
        return;
    }

    for (const auto& prod : products) {
        if (std::find(available.begin(), available.end(), prod) == available.end()) {
            *err << err_colors.error << "error: " << err_colors.reset
                 << "product '" << prod << "' not found in physics products\n";
            *err << "available products: ";
            auto first = true;
            for (const auto& name : available) {
                if (!first) *err << ", ";
                *err << name;
                first = false;
            }
            *err << "\n";
            return;
        }
    }

    prog.driver_state.selected_products = products;
    handle_show_products();
}

template<Physics P>
void driver_t<P>::handle_clear_products() {
    prog.driver_state.selected_products.clear();
    *out << colors.info << "clear product selection" << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_do_timeseries() {
    if (!prog.physics_state.has_value()) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "physics state not initialized; use 'init' command first\n";
        return;
    }
    if (prog.driver_state.timeseries.empty()) {
        *err << err_colors.warning << "no timeseries selected; " << err_colors.reset
             << "use 'select timeseries [names...]'\n";
        return;
    }
    *out << colors.info << "record timeseries sample" << colors.reset << " (";
    auto first = true;
    for (const auto& [col, values] : prog.driver_state.timeseries) {
        const auto value = get_timeseries(prog.physics, prog.initial, *prog.physics_state, col);
        prog.driver_state.timeseries[col].push_back(value);
        if (!first) *out << ", ";
        *out << colors.label << col << "=" << colors.reset
             << colors.value << std::scientific << std::showpos << std::setprecision(6)
             << value << std::noshowpos << colors.reset;
        first = false;
    }
    *out << ")\n";
}

template<Physics P>
void driver_t<P>::handle_write_timeseries(const std::optional<std::string>& filename_opt) {
    if (prog.driver_state.timeseries.empty()) {
        *err << err_colors.warning << "no timeseries selected; " << err_colors.reset
             << "use 'select timeseries [names...]'\n";
        return;
    }

    auto filename = std::string{};
    auto format = driver::output_format{};

    if (filename_opt) {
        filename = *filename_opt;
        format = driver::infer_format_from_filename(filename);
    } else {
        format = prog.driver_state.format;
        const auto ext = (format == driver::output_format::ascii) ? ".dat" : ".bin";
        auto oss = std::ostringstream{};
        oss << "timeseries." << std::setw(4) << std::setfill('0') << prog.driver_state.timeseries_count << ext;
        filename = oss.str();
        prog.driver_state.timeseries_count++;
    }

    if (format == driver::output_format::hdf5) {
        throw std::runtime_error("HDF5 output format not implemented");
    }

    auto file = std::ofstream{filename};
    if (!file) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "failed to open " << filename << "\n";
        return;
    }
    if (format == driver::output_format::ascii) {
        auto writer = ascii_writer{file};
        serialize(writer, "timeseries", prog.driver_state.timeseries);
    } else {
        auto writer = binary_writer{file};
        serialize(writer, "timeseries", prog.driver_state.timeseries);
    }
    *out << colors.info << "write " << colors.value << filename << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_write_checkpoint(const std::optional<std::string>& filename_opt) {
    auto filename = std::string{};
    auto format = driver::output_format{};

    if (filename_opt) {
        filename = *filename_opt;
        format = driver::infer_format_from_filename(filename);
    } else {
        format = prog.driver_state.format;
        const auto ext = (format == driver::output_format::ascii) ? ".dat" : ".bin";
        auto oss = std::ostringstream{};
        oss << "chkpt." << std::setw(4) << std::setfill('0') << prog.driver_state.checkpoint_count << ext;
        filename = oss.str();
        prog.driver_state.checkpoint_count++;
    }

    if (format == driver::output_format::hdf5) {
        throw std::runtime_error("HDF5 output format not implemented");
    }

    auto file = std::ofstream{filename};
    if (format == driver::output_format::ascii) {
        auto writer = ascii_writer{file};
        serialize(writer, "checkpoint", prog);
    } else {
        auto writer = binary_writer{file};
        serialize(writer, "checkpoint", prog);
    }
    *out << colors.info << "write " << colors.value << filename << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_write_products(const std::optional<std::string>& filename_opt) {
    if (!prog.physics_state.has_value()) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "physics state not initialized; use 'init' command first\n";
        return;
    }
    if (prog.driver_state.selected_products.empty()) {
        *err << err_colors.warning << "no products selected; " << err_colors.reset
             << "use 'select products [names...]'\n";
        return;
    }

    auto filename = std::string{};
    auto format = driver::output_format{};

    if (filename_opt) {
        filename = *filename_opt;
        format = driver::infer_format_from_filename(filename);
    } else {
        format = prog.driver_state.format;
        const auto ext = (format == driver::output_format::ascii) ? ".dat" : ".bin";
        auto oss = std::ostringstream{};
        oss << "prods." << std::setw(4) << std::setfill('0') << prog.driver_state.products_count << ext;
        filename = oss.str();
        prog.driver_state.products_count++;
    }

    if (format == driver::output_format::hdf5) {
        throw std::runtime_error("HDF5 output format not implemented");
    }

    auto file = std::ofstream{filename};

    if (format == driver::output_format::ascii) {
        auto writer = ascii_writer{file};
        for (const auto& name : prog.driver_state.selected_products) {
            const auto product = get_product(*prog.physics_state, name, exec_context);
            serialize(writer, name.c_str(), product);
        }
    } else {
        auto writer = binary_writer{file};
        for (const auto& name : prog.driver_state.selected_products) {
            const auto product = get_product(*prog.physics_state, name, exec_context);
            serialize(writer, name.c_str(), product);
        }
    }
    *out << colors.info << "write " << colors.value << filename << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_show_message() {
    *out << iteration_message();
}

template<Physics P>
void driver_t<P>::handle_repeat_add(double interval, const std::string& unit, const std::string& sub_cmd) {
    auto rcmd = driver::recurring_command_t{};
    rcmd.interval = interval;
    rcmd.unit = unit;
    rcmd.sub_command = sub_cmd;
    prog.driver_state.recurring_commands.push_back(rcmd);
    *out << colors.info << "add recurring command: " << colors.reset
         << "every " << colors.value << interval << colors.reset << " " << unit
         << " -> " << colors.value << sub_cmd << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_repeat_list() {
    if (prog.driver_state.recurring_commands.empty()) {
        *out << colors.unselected << "no recurring commands." << colors.reset << "\n";
    } else {
        *out << colors.header << "recurring commands:" << colors.reset << "\n";
        for (std::size_t i = 0; i < prog.driver_state.recurring_commands.size(); ++i) {
            const auto& rcmd = prog.driver_state.recurring_commands[i];
            *out << "  " << colors.iteration << "[" << i << "]" << colors.reset
                 << " every " << colors.value << rcmd.interval << colors.reset << " " << rcmd.unit
                 << " -> " << colors.value << rcmd.sub_command << colors.reset << "\n";
        }
    }
}

template<Physics P>
void driver_t<P>::handle_repeat_clear(const std::vector<std::string>& indices) {
    if (indices.empty()) {
        prog.driver_state.recurring_commands.clear();
        *out << colors.info << "clear all recurring commands" << colors.reset << "\n";
    } else {
        auto idx_list = std::vector<std::size_t>{};
        for (const auto& idx_str : indices) {
            try {
                const auto idx = std::stoull(idx_str);
                if (idx >= prog.driver_state.recurring_commands.size()) {
                    *err << err_colors.error << "error: " << err_colors.reset
                         << "index " << idx << " out of range (max: "
                         << prog.driver_state.recurring_commands.size() - 1 << ")\n";
                    continue;
                }
                idx_list.push_back(idx);
            } catch (...) {
                *err << err_colors.error << "error: " << err_colors.reset
                     << "invalid index: " << idx_str << "\n";
            }
        }
        std::sort(idx_list.begin(), idx_list.end(), std::greater<std::size_t>());
        idx_list.erase(std::unique(idx_list.begin(), idx_list.end()), idx_list.end());
        for (const auto idx : idx_list) {
            prog.driver_state.recurring_commands.erase(
                prog.driver_state.recurring_commands.begin() + idx);
        }
        *out << colors.info << "clear " << colors.value << idx_list.size()
             << colors.reset << colors.info << " recurring command(s)" << colors.reset << "\n";
    }
}

template<Physics P>
void driver_t<P>::handle_init() {
    if (prog.physics_state.has_value()) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "state already initialized; use 'reset' to clear state first\n";
        return;
    }
    prog.physics_state = initial_state(exec_context);
    *out << colors.info << "initialized physics state" << colors.reset << "\n";
}

template<Physics P>
void driver_t<P>::handle_reset() {
    prog.physics_state = std::nullopt;
    prog.driver_state = driver::state_t{};
    *out << colors.info << "cleared physics state" << colors.reset << "\n";
}

template<typename T>
void deserialize_with_format(std::ifstream& file, driver::output_format format, const char* name, T& value) {
    if (format == driver::output_format::ascii) {
        auto reader = ascii_reader{file};
        deserialize(reader, name, value);
    } else {
        auto reader = binary_reader{file};
        deserialize(reader, name, value);
    }
}

template<Physics P>
void driver_t<P>::handle_load(const std::string& filename) {
    // Check for command sequence files (.prog or .mist)
    if (filename.ends_with(".prog") || filename.ends_with(".mist")) {
        auto file = std::ifstream{filename};
        if (!file) {
            *err << err_colors.error << "error: " << err_colors.reset
                 << "failed to open " << filename << "\n";
            return;
        }
        *out << colors.info << "loading commands from " << colors.value << filename
             << colors.reset << "\n";
        auto line = std::string{};
        auto line_num = 0;
        while (std::getline(file, line)) {
            ++line_num;
            // Trim leading whitespace
            const auto start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            line = line.substr(start);
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;
            // Echo the command
            *out << colors.prompt << "> " << colors.reset << line << "\n";
            // Parse and execute
            const auto cmd = parse_command(line);
            if (cmd.cmd == command_t::type::invalid) {
                *err << err_colors.error << filename << ":" << line_num << ": error: "
                     << err_colors.reset << cmd.error_msg << "\n";
                continue;
            }
            if (!execute(cmd)) {
                // Stop command was encountered
                return;
            }
        }
        *out << colors.info << "finished loading " << colors.value << filename
             << colors.reset << "\n";
        return;
    }

    auto file = std::ifstream{filename};
    if (!file) {
        *err << err_colors.error << "error: " << err_colors.reset
             << "failed to open " << filename << "\n";
        return;
    }

    const auto format = driver::infer_format_from_filename(filename);
    if (format == driver::output_format::hdf5) {
        throw std::runtime_error("HDF5 format not implemented");
    }

    // Try checkpoint first - if successful, we're done
    try {
        deserialize_with_format(file, format, "checkpoint", prog);
        *out << colors.info << "loaded checkpoint from " << colors.value << filename
             << colors.reset << "\n";
        return;
    } catch (...) {
        // Not a checkpoint, reopen and continue
        file.close();
        file.open(filename);
    }

    // Try loading physics
    try {
        deserialize_with_format(file, format, "physics", prog.physics);
        prog.physics_state = std::nullopt;
        prog.driver_state.iteration = 0;
        *out << colors.info << "loaded physics from " << colors.value << filename
             << colors.reset << "; use 'init' to generate initial state\n";
        return;
    } catch (...) {
        // Not physics, reopen and continue
        file.close();
        file.open(filename);
    }

    // Try loading initial
    try {
        deserialize_with_format(file, format, "initial", prog.initial);
        if (prog.physics_state.has_value()) {
            *err << err_colors.warning << "warning: " << err_colors.reset
                 << "loaded initial but state is already initialized; use 'reset' first\n";
        }
        *out << colors.info << "loaded initial from " << colors.value << filename
             << colors.reset << "\n";
        return;
    } catch (...) {
        // Not initial either
    }

    *err << err_colors.error << "error: " << err_colors.reset
         << "could not load checkpoint, physics, or initial from " << filename << "\n";
}

template<Physics P>
void driver_t<P>::handle_show_all() {
    handle_show_message();
    *out << "\n";
    handle_show_initial();
    *out << "\n";
    handle_show_physics();
    *out << "\n";
    handle_show_driver();
    *out << "\n";
    handle_show_products();
    *out << "\n";
    handle_show_timeseries();
    *out << "\n";
}

template<Physics P>
void driver_t<P>::handle_show_physics() {
    auto writer = ascii_writer{*out};
    serialize(writer, "physics", prog.physics);
}

template<Physics P>
void driver_t<P>::handle_show_initial() {
    auto writer = ascii_writer{*out};
    serialize(writer, "initial", prog.initial);
}

template<Physics P>
void driver_t<P>::handle_show_timeseries() {
    const auto available = names_of_timeseries(std::type_identity<P>{});
    *out << colors.header << "Timeseries:" << colors.reset << "\n";
    for (const auto& col : available) {
        const auto is_selected = prog.driver_state.timeseries.find(col) != prog.driver_state.timeseries.end();
        if (is_selected) {
            *out << "  - " << colors.selected << "[+]" << colors.reset << " "
                 << colors.key << col << colors.reset;
            const auto sample_count = prog.driver_state.timeseries.at(col).size();
            *out << " (" << colors.value << sample_count << colors.reset
                 << " sample" << (sample_count != 1 ? "s" : "") << ")";
        } else {
            *out << "  - " << colors.unselected << "[ ] " << col << colors.reset;
        }
        *out << "\n";
    }
}

template<Physics P>
void driver_t<P>::handle_show_products() {
    const auto available = names_of_products(std::type_identity<P>{});
    *out << colors.header << "Products:" << colors.reset << "\n";
    for (const auto& prod : available) {
        const auto is_selected = std::find(
            prog.driver_state.selected_products.begin(),
            prog.driver_state.selected_products.end(),
            prod
        ) != prog.driver_state.selected_products.end();
        if (is_selected) {
            *out << "  - " << colors.selected << "[+]" << colors.reset << " "
                 << colors.key << prod << colors.reset << "\n";
        } else {
            *out << "  - " << colors.unselected << "[ ] " << prod << colors.reset << "\n";
        }
    }
}

template<Physics P>
void driver_t<P>::handle_show_driver() {
    auto writer = ascii_writer{*out};
    serialize(writer, "driver_state", prog.driver_state);
}

template<Physics P>
void driver_t<P>::handle_help() {
    *out << "\n" << help_text << "\n";
}

template<Physics P>
void driver_t<P>::handle_stop() {
    *out << "\n" << colors.header << "=== Simulation Complete ===" << colors.reset << "\n";
    if (prog.physics_state.has_value()) {
        const auto time_names = names_of_time(std::type_identity<P>{});
        *out << "Final times: ";
        for (std::size_t i = 0; i < time_names.size(); ++i) {
            if (i > 0) *out << " ";
            *out << colors.label << time_names[i] << "=" << colors.reset
                 << colors.value << std::scientific << std::showpos << std::setprecision(6)
                 << get_time(*prog.physics_state, time_names[i]) << std::noshowpos << colors.reset;
        }
        *out << "\n";
    }
}

template<Physics P>
bool driver_t<P>::execute(const command_t& cmd) {
    try {
        command_start_wall_time = get_wall_time();
        command_start_iteration = prog.driver_state.iteration;

        switch (cmd.cmd) {
            case command_t::type::increment_n:
                handle_increment_n(cmd.int_value.value());
                break;

            case command_t::type::target_n:
                handle_target_n(cmd.int_value.value());
                break;

            case command_t::type::increment_var:
                handle_increment_var(cmd.var_name.value(), cmd.double_value.value());
                break;

            case command_t::type::target_var:
                handle_target_var(cmd.var_name.value(), cmd.double_value.value());
                break;

            case command_t::type::set_output:
                handle_set_output(cmd.string_value.value());
                break;

            case command_t::type::set_physics:
                handle_set_physics(cmd.string_value.value());
                break;

            case command_t::type::set_initial:
                handle_set_initial(cmd.string_value.value());
                break;

            case command_t::type::set_exec:
                handle_set_exec(cmd.string_value.value());
                break;

            case command_t::type::clear_timeseries:
                handle_clear_timeseries();
                break;

            case command_t::type::select_timeseries:
                handle_select_timeseries(cmd.string_list.value());
                break;

            case command_t::type::clear_products:
                handle_clear_products();
                break;

            case command_t::type::select_products:
                handle_select_products(cmd.string_list.value_or(std::vector<std::string>{}));
                break;

            case command_t::type::do_timeseries:
                handle_do_timeseries();
                break;

            case command_t::type::write_timeseries:
                handle_write_timeseries(cmd.string_value);
                break;

            case command_t::type::write_checkpoint:
                handle_write_checkpoint(cmd.string_value);
                break;

            case command_t::type::write_products:
                handle_write_products(cmd.string_value);
                break;

            case command_t::type::show_message:
                handle_show_message();
                break;

            case command_t::type::repeat_add:
                handle_repeat_add(cmd.double_value.value(), cmd.var_name.value(),
                                 cmd.string_value.value());
                break;

            case command_t::type::repeat_list:
                handle_repeat_list();
                break;

            case command_t::type::repeat_clear:
                handle_repeat_clear(cmd.string_list.value_or(std::vector<std::string>{}));
                break;

            case command_t::type::init:
                handle_init();
                break;

            case command_t::type::reset:
                handle_reset();
                break;

            case command_t::type::load:
                handle_load(cmd.string_value.value());
                break;

            case command_t::type::show_all:
                handle_show_all();
                break;

            case command_t::type::show_physics:
                handle_show_physics();
                break;

            case command_t::type::show_initial:
                handle_show_initial();
                break;

            case command_t::type::show_timeseries:
                handle_show_timeseries();
                break;

            case command_t::type::show_products:
                handle_show_products();
                break;

            case command_t::type::show_driver:
                handle_show_driver();
                break;

            case command_t::type::help:
                handle_help();
                break;

            case command_t::type::stop:
                handle_stop();
                return false;

            case command_t::type::invalid:
                break;
        }
    } catch (const std::exception& e) {
        *err << err_colors.error << "Error: " << err_colors.reset << e.what() << "\n";
    }
    return true;
}

// =============================================================================
// REPL class - handles interactive input/output
// =============================================================================

template<Physics P>
class repl_t {
    driver_t<P>& driver;
    std::queue<std::string> command_queue;
    bool is_tty;
    FILE* null_stream = nullptr;

public:
    explicit repl_t(driver_t<P>& driver)
        : driver(driver)
        , is_tty(isatty(STDIN_FILENO)) {
        setup_readline();
    }

    ~repl_t() {
        if (null_stream) {
            fclose(null_stream);
        }
    }

    void run();

private:
    void setup_readline();
    // void print_initial_state();
    auto get_next_command() -> std::optional<std::string>;
    void switch_to_interactive_mode();
};

template<Physics P>
void repl_t<P>::setup_readline() {
    if (!is_tty) {
        null_stream = fopen("/dev/null", "w");
        rl_outstream = null_stream;
    }
}

// template<Physics P>
// void repl_t<P>::print_initial_state() {
//     ascii_writer cfg_writer(std::cout);
//     serialize(cfg_writer, "initial", driver.program().initial);
//     std::cout << "\n";
//     serialize(cfg_writer, "physics", driver.program().physics);
//     if (is_tty) {
//         std::cout << "\nType 'help' for available commands\n";
//     }
// }

template<Physics P>
std::optional<std::string> repl_t<P>::get_next_command() {
    if (!command_queue.empty()) {
        auto cmd = command_queue.front();
        command_queue.pop();
        return cmd;
    }

    auto accumulated = std::string{};
    const auto* prompt = is_tty ? "> " : "";

    while (true) {
        auto* line = readline(prompt);
        if (!line) {
            if (!accumulated.empty()) {
                return accumulated;
            }
            return std::nullopt;
        }

        auto input = std::string{line};
        std::free(line);

        if (!is_tty && !input.empty() && input[0] != '#') {
            std::cout << "> " << input << "\n";
        }

        auto has_continuation = false;
        if (!input.empty() && input.back() == '\\') {
            has_continuation = true;
            input.pop_back();
            while (!input.empty() && (input.back() == ' ' || input.back() == '\t')) {
                input.pop_back();
            }
        }

        if (!accumulated.empty()) {
            accumulated += " ";
        }
        accumulated += input;

        if (!has_continuation) {
            auto iss = std::istringstream{accumulated};
            auto first_line = std::string{};
            if (std::getline(iss, first_line)) {
                auto remaining_line = std::string{};
                while (std::getline(iss, remaining_line)) {
                    if (!remaining_line.empty()) {
                        command_queue.push(remaining_line);
                    }
                }
                return first_line;
            }
            return accumulated;
        }

        prompt = is_tty ? "... " : "";
    }
}

template<Physics P>
void repl_t<P>::switch_to_interactive_mode() {
    std::cout << "\n=== Piped input complete, entering interactive mode ===\n\n";
    std::cout << "Type 'help' for available commands\n";
    is_tty = true;
    if (null_stream) {
        fclose(null_stream);
        null_stream = nullptr;
    }
    rl_outstream = stderr;
    if (freopen("/dev/tty", "r", stdin) == nullptr) {
        throw std::runtime_error("failed to reopen stdin as tty");
    }
}

template<Physics P>
void repl_t<P>::run() {
    // print_initial_state();

    while (true) {
        const auto input_opt = get_next_command();
        if (!input_opt) {
            if (!is_tty) {
                try {
                    switch_to_interactive_mode();
                    continue;
                } catch (...) {
                    std::cout << "\n";
                    break;
                }
            }
            std::cout << "\n";
            break;
        }

        const auto input = *input_opt;

        if (input.empty() || input[0] == '#') {
            continue;
        }

        add_history(input.c_str());

        const auto cmd = parse_command(input);

        if (cmd.cmd == command_t::type::invalid) {
            const auto err_colors = color::auto_scheme(std::cerr);
            std::cerr << err_colors.error << "error: " << err_colors.reset << cmd.error_msg << "\n";
            continue;
        }

        if (!driver.execute(cmd)) {
            break;
        }
    }
}

// =============================================================================
// Main run function (backward compatibility wrapper)
// =============================================================================

template<Physics P>
std::optional<typename P::state_t> run(program_t<P>& prog)
{
    auto driver = driver_t<P>{prog, std::cout, std::cerr};
    auto repl = repl_t<P>{driver};
    repl.run();
    return std::move(prog.physics_state);
}

} // namespace mist
