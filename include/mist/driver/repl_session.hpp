#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <queue>
#include <sstream>
#include <unistd.h>
#include <readline/history.h>
#include <readline/readline.h>
#include "engine.hpp"

namespace mist::driver {

// =============================================================================
// ANSI Color Support
// =============================================================================

namespace color {

namespace ansi {
    inline constexpr const char* reset      = "\033[0m";
    inline constexpr const char* bold       = "\033[1m";
    inline constexpr const char* dim        = "\033[2m";
    inline constexpr const char* black      = "\033[30m";
    inline constexpr const char* red        = "\033[31m";
    inline constexpr const char* green      = "\033[32m";
    inline constexpr const char* yellow     = "\033[33m";
    inline constexpr const char* blue       = "\033[34m";
    inline constexpr const char* magenta    = "\033[35m";
    inline constexpr const char* cyan       = "\033[36m";
    inline constexpr const char* white      = "\033[37m";
    inline constexpr const char* bright_white  = "\033[97m";
    inline constexpr const char* bright_cyan   = "\033[96m";
} // namespace ansi

struct scheme_t {
    const char* reset       = ansi::reset;
    const char* iteration   = ansi::cyan;
    const char* label       = ansi::blue;
    const char* value       = ansi::bright_white;
    const char* info        = ansi::green;
    const char* warning     = ansi::yellow;
    const char* error       = ansi::red;
    const char* prompt      = ansi::bright_cyan;
    const char* key         = ansi::magenta;
    const char* selected    = ansi::green;
    const char* unselected  = ansi::dim;
    const char* header      = ansi::bold;
};

inline auto enabled_scheme() -> scheme_t { return scheme_t{}; }
inline auto disabled_scheme() -> scheme_t {
    return scheme_t{"", "", "", "", "", "", "", "", "", "", "", ""};
}

inline auto is_tty(int fd) -> bool { return isatty(fd) != 0; }
inline auto is_tty(std::ostream& os) -> bool {
    if (&os == &std::cout) return is_tty(STDOUT_FILENO);
    if (&os == &std::cerr) return is_tty(STDERR_FILENO);
    return false;
}

inline auto auto_scheme(std::ostream& os) -> scheme_t {
    return is_tty(os) ? enabled_scheme() : disabled_scheme();
}

} // namespace color

// =============================================================================
// Command parsing
// =============================================================================

struct parsed_command_t {
    std::optional<command_t> cmd;
    std::optional<cmd::repeat_add> repeat;
    std::string error;
};

auto parse_command(std::string_view input) -> parsed_command_t;

// =============================================================================
// repl_session_t - interactive REPL session
// =============================================================================

class repl_session_t {
public:
    repl_session_t(engine_t& engine, std::ostream& out = std::cout, std::ostream& err = std::cerr);
    ~repl_session_t();

    void run();

private:
    engine_t& engine_;
    std::ostream& out_;
    std::ostream& err_;
    color::scheme_t colors_;
    color::scheme_t err_colors_;
    std::queue<std::string> command_queue_;
    bool is_tty_;
    FILE* null_stream_ = nullptr;
    bool should_stop_ = false;
    bool had_error_ = false;

    void setup_readline();
    auto get_next_command() -> std::optional<std::string>;
    void switch_to_interactive_mode();
    auto execute_line(const std::string& input) -> bool;
    auto load_script(const std::string& filename) -> bool;
    void show_recurring_commands();
    void format_response(const response_t& r);

    void format(const resp::ok& r);
    void format(const resp::error& r);
    void format(const resp::interrupted& r);
    void format(const resp::stopped& r);
    void format(const resp::state_info& r);
    void format(const resp::iteration_status& r);
    void format(const resp::timeseries_sample& r);
    void format(const resp::physics_config& r);
    void format(const resp::initial_config& r);
    void format(const resp::driver_state& r);
    void format(const resp::help_text& r);
    void format(const resp::timeseries_info& r);
    void format(const resp::products_info& r);
    void format(const resp::profiler_info& r);
    void format(const resp::wrote_file& r);
};

} // namespace mist::driver

// =============================================================================
// Include implementations for header-only mode
// =============================================================================

#ifndef MIST_DRIVER_SEPARATE_COMPILATION
#include "repl_session.ipp"
#endif
