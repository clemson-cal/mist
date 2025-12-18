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
    void show_repeating_commands();
    void format_response(const response_t& r);
};

} // namespace mist::driver
