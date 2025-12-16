// repl_session.ipp - implementation file for repl_session_t

#ifdef MIST_DRIVER_SEPARATE_COMPILATION
#define MIST_INLINE
#else
#define MIST_INLINE inline
#endif

namespace mist::driver {

// =============================================================================
// Command parsing
// =============================================================================

MIST_INLINE auto parse_command(std::string_view input) -> parsed_command_t {
    auto iss = std::istringstream{std::string{input}};
    auto first = std::string{};
    iss >> first;

    if (first.empty()) {
        return {{}, {}, "empty command"};
    }

    // n++ or n+=X or n->X
    if (first == "n++" || first == "n") {
        if (first == "n++") {
            return {cmd::advance_by{"n", 1.0}, {}, {}};
        }
        auto op = std::string{};
        iss >> op;
        auto val = 0.0;
        if (!(iss >> val)) {
            return {{}, {}, "n " + op + " requires numeric value"};
        }
        if (op == "+=") {
            return {cmd::advance_by{"n", val}, {}, {}};
        } else if (op == "->") {
            return {cmd::advance_to{"n", val}, {}, {}};
        }
        return {{}, {}, "unknown operator: " + op};
    }

    // VAR += X or VAR -> X
    if (input.find("+=") != std::string_view::npos || input.find("->") != std::string_view::npos) {
        auto var_name = first;
        auto op = std::string{};
        auto val = 0.0;
        iss >> op >> val;
        if (op == "+=") {
            return {cmd::advance_by{var_name, val}, {}, {}};
        } else if (op == "->") {
            return {cmd::advance_to{var_name, val}, {}, {}};
        }
    }

    // set
    if (first == "set") {
        auto what = std::string{};
        iss >> what;

        if (what == "physics" || what == "initial" || what == "exec") {
            auto key = std::string{};
            auto value = std::string{};
            if (!(iss >> key)) {
                return {{}, {}, "set " + what + " requires key=value"};
            }
            auto eq = key.find('=');
            if (eq == std::string::npos) {
                return {{}, {}, "set " + what + " requires key=value format"};
            }
            value = key.substr(eq + 1);
            key = key.substr(0, eq);

            if (what == "physics") return {cmd::set_physics{key, value}, {}, {}};
            if (what == "initial") return {cmd::set_initial{key, value}, {}, {}};
            if (what == "exec") return {cmd::set_exec{key, value}, {}, {}};
        }

        auto eq = what.find('=');
        if (eq != std::string::npos) {
            auto key = what.substr(0, eq);
            auto value = what.substr(eq + 1);
            if (key == "output") {
                return {cmd::set_output{value}, {}, {}};
            }
            return {{}, {}, "unknown setting: " + key};
        }
        return {{}, {}, "set requires format: set key=value or set physics/initial/exec key=value"};
    }

    // select
    if (first == "select") {
        auto what = std::string{};
        iss >> what;
        auto items = std::vector<std::string>{};
        auto item = std::string{};
        while (iss >> item) items.push_back(item);

        if (what == "timeseries") return {cmd::select_timeseries{items}, {}, {}};
        if (what == "products") return {cmd::select_products{items}, {}, {}};
        return {{}, {}, "unknown: select " + what};
    }

    // clear
    if (first == "clear") {
        auto what = std::string{};
        iss >> what;
        if (what == "repeat") return {cmd::clear_repeat{}, {}, {}};
        return {{}, {}, "unknown: clear " + what};
    }

    // do
    if (first == "do") {
        auto what = std::string{};
        iss >> what;
        if (what == "timeseries") return {cmd::do_timeseries{}, {}, {}};
        return {{}, {}, "unknown: do " + what};
    }

    // write
    if (first == "write") {
        auto what = std::string{};
        iss >> what;

        auto dest = std::string{};
        auto has_dest = static_cast<bool>(iss >> dest);

        if (what == "physics") {
            if (!has_dest) return {{}, {}, "write physics requires destination"};
            return {cmd::write_physics{dest}, {}, {}};
        }
        if (what == "initial") {
            if (!has_dest) return {{}, {}, "write initial requires destination"};
            return {cmd::write_initial{dest}, {}, {}};
        }
        if (what == "driver") {
            if (!has_dest) return {{}, {}, "write driver requires destination"};
            return {cmd::write_driver{dest}, {}, {}};
        }
        if (what == "profiler") {
            if (!has_dest) return {{}, {}, "write profiler requires destination"};
            return {cmd::write_profiler{dest}, {}, {}};
        }
        if (what == "timeseries") {
            return {cmd::write_timeseries{has_dest ? std::optional{dest} : std::nullopt}, {}, {}};
        }
        if (what == "checkpoint") {
            return {cmd::write_checkpoint{has_dest ? std::optional{dest} : std::nullopt}, {}, {}};
        }
        if (what == "products") {
            return {cmd::write_products{has_dest ? std::optional{dest} : std::nullopt}, {}, {}};
        }
        if (what == "iteration") {
            return {cmd::write_iteration{has_dest ? std::optional{dest} : std::nullopt}, {}, {}};
        }
        return {{}, {}, "unknown: write " + what};
    }

    // repeat
    if (first == "repeat") {
        auto interval_str = std::string{};
        iss >> interval_str;

        if (interval_str == "list") {
            // This is a show command, handled by session
            return {{}, {}, "repeat list"};
        }

        auto interval = 0.0;
        try {
            interval = std::stod(interval_str);
        } catch (...) {
            return {{}, {}, "repeat requires numeric interval"};
        }

        auto unit = std::string{};
        if (!(iss >> unit)) {
            return {{}, {}, "repeat requires unit"};
        }

        auto sub_cmd_str = std::string{};
        std::getline(iss, sub_cmd_str);
        auto start = sub_cmd_str.find_first_not_of(" \t");
        if (start == std::string::npos) {
            return {{}, {}, "repeat requires sub-command"};
        }
        sub_cmd_str = sub_cmd_str.substr(start);

        auto sub_parsed = parse_command(sub_cmd_str);
        if (!sub_parsed.cmd) {
            return {{}, {}, "invalid sub-command: " + sub_parsed.error};
        }

        return {{}, cmd::repeat_add{interval, unit, *sub_parsed.cmd}, {}};
    }

    // load
    if (first == "load") {
        auto filename = std::string{};
        if (!(iss >> filename)) {
            return {{}, {}, "load requires filename"};
        }
        return {cmd::load{filename}, {}, {}};
    }

    // init / reset
    if (first == "init") return {cmd::init{}, {}, {}};
    if (first == "reset") return {cmd::reset{}, {}, {}};

    // show
    if (first == "show") {
        auto what = std::string{};
        iss >> what;
        if (what.empty() || what == "all") return {cmd::show_all{}, {}, {}};
        if (what == "state") return {cmd::show_state{}, {}, {}};
        if (what == "physics") return {cmd::show_physics{}, {}, {}};
        if (what == "initial") return {cmd::show_initial{}, {}, {}};
        if (what == "iteration") return {cmd::show_iteration{}, {}, {}};
        if (what == "timeseries") return {cmd::show_timeseries{}, {}, {}};
        if (what == "products") return {cmd::show_products{}, {}, {}};
        if (what == "profiler") return {cmd::show_profiler{}, {}, {}};
        if (what == "driver") return {cmd::show_driver{}, {}, {}};
        return {{}, {}, "unknown: show " + what};
    }

    // help / stop
    if (first == "help") {
        auto what = std::string{};
        iss >> what;
        if (what == "schema") return {cmd::help_schema{}, {}, {}};
        return {cmd::help{}, {}, {}};
    }
    if (first == "stop" || first == "quit" || first == "q") return {cmd::stop{}, {}, {}};

    return {{}, {}, "unknown command: " + first};
}

// =============================================================================
// repl_session_t implementation
// =============================================================================

MIST_INLINE repl_session_t::repl_session_t(engine_t& engine, std::ostream& out, std::ostream& err)
    : engine_(engine)
    , out_(out)
    , err_(err)
    , colors_(color::for_stream(out))
    , err_colors_(color::for_stream(err))
    , is_tty_(isatty(STDIN_FILENO))
{
    setup_readline();
    if (!is_tty_) {
        out_.setf(std::ios::unitbuf);
    }
}

MIST_INLINE repl_session_t::~repl_session_t() {
    if (null_stream_) {
        fclose(null_stream_);
    }
}

MIST_INLINE void repl_session_t::run() {
    while (true) {
        auto input_opt = get_next_command();
        if (!input_opt) {
            if (!is_tty_) {
                try {
                    switch_to_interactive_mode();
                    continue;
                } catch (...) {
                    out_ << "\n";
                    break;
                }
            }
            out_ << "\n";
            break;
        }

        auto input = *input_opt;
        if (input.empty() || input[0] == '#') continue;

        add_history(input.c_str());

        if (!execute_line(input)) {
            break;
        }
    }
}

MIST_INLINE void repl_session_t::setup_readline() {
    if (!is_tty_) {
        null_stream_ = fopen("/dev/null", "w");
        rl_outstream = null_stream_;
    }
}

MIST_INLINE auto repl_session_t::get_next_command() -> std::optional<std::string> {
    if (!command_queue_.empty()) {
        auto cmd = command_queue_.front();
        command_queue_.pop();
        return cmd;
    }

    auto accumulated = std::string{};
    auto prompt = is_tty_ ? "> " : "";

    while (true) {
        auto* line = readline(prompt);
        if (!line) {
            if (!accumulated.empty()) return accumulated;
            return std::nullopt;
        }

        auto input = std::string{line};
        std::free(line);

        if (!is_tty_ && !input.empty() && input[0] != '#') {
            out_ << "> " << input << "\n";
        }

        auto has_continuation = false;
        if (!input.empty() && input.back() == '\\') {
            has_continuation = true;
            input.pop_back();
            while (!input.empty() && (input.back() == ' ' || input.back() == '\t')) {
                input.pop_back();
            }
        }

        if (!accumulated.empty()) accumulated += " ";
        accumulated += input;

        if (!has_continuation) {
            auto iss = std::istringstream{accumulated};
            auto first_line = std::string{};
            if (std::getline(iss, first_line)) {
                auto remaining = std::string{};
                while (std::getline(iss, remaining)) {
                    if (!remaining.empty()) command_queue_.push(remaining);
                }
                return first_line;
            }
            return accumulated;
        }

        prompt = is_tty_ ? "... " : "";
    }
}

MIST_INLINE void repl_session_t::switch_to_interactive_mode() {
    out_ << "\n=== Piped input complete, entering interactive mode ===\n\n";
    out_ << "Type 'help' for available commands\n";
    is_tty_ = true;
    if (null_stream_) {
        fclose(null_stream_);
        null_stream_ = nullptr;
    }
    rl_outstream = stderr;
    if (freopen("/dev/tty", "r", stdin) == nullptr) {
        throw std::runtime_error("failed to reopen stdin as tty");
    }
}

MIST_INLINE auto repl_session_t::execute_line(const std::string& input) -> bool {
    // Handle script files
    if (input.ends_with(".prog") || input.ends_with(".mist")) {
        return load_script(input);
    }

    auto parsed = parse_command(input);

    // Handle special "repeat list" case
    if (parsed.error == "repeat list") {
        show_repeating_commands();
        return true;
    }

    if (!parsed.cmd && !parsed.repeat) {
        err_ << err_colors_.error << "error: " << err_colors_.reset << parsed.error << "\n";
        return is_tty_;  // Continue if interactive, stop if piped
    }

    had_error_ = false;
    should_stop_ = false;

    auto emit = [this](const response_t& r) { format_response(r); };

    if (parsed.repeat) {
        engine_.execute(*parsed.repeat, emit);
    } else {
        engine_.execute(*parsed.cmd, emit);
    }

    if (had_error_ && !is_tty_) return false;
    return !should_stop_;
}

MIST_INLINE auto repl_session_t::load_script(const std::string& filename) -> bool {
    auto file = std::ifstream{filename};
    if (!file) {
        err_ << err_colors_.error << "error: " << err_colors_.reset
             << "failed to open " << filename << "\n";
        return is_tty_;
    }

    out_ << colors_.info << "loading " << colors_.value << filename
         << colors_.reset << "\n";

    auto line = std::string{};
    auto line_num = 0;
    while (std::getline(file, line)) {
        ++line_num;
        auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.empty() || line[0] == '#') continue;

        out_ << colors_.prompt << "> " << colors_.reset << line << "\n";

        if (!execute_line(line)) {
            if (had_error_) {
                err_ << err_colors_.error << "error at " << err_colors_.reset
                     << filename << ":" << line_num << "\n";
            }
            return false;
        }
    }

    out_ << colors_.info << "finished loading " << colors_.value << filename
         << colors_.reset << "\n";
    return true;
}

MIST_INLINE void repl_session_t::show_repeating_commands() {
    auto& state = engine_.state();
    if (state.repeating_commands.empty()) {
        out_ << colors_.unselected << "no repeating commands" << colors_.reset << "\n";
        return;
    }

    out_ << colors_.header << "repeating commands:" << colors_.reset << "\n";
    for (std::size_t i = 0; i < state.repeating_commands.size(); ++i) {
        const auto& rc = state.repeating_commands[i];
        out_ << "  " << colors_.iteration << "[" << i << "]" << colors_.reset
             << " every " << colors_.value << rc.interval << colors_.reset
             << " " << rc.unit << "\n";
    }
}

MIST_INLINE void repl_session_t::format_response(const response_t& r) {
    std::visit([this](const auto& resp) {
        using T = std::decay_t<decltype(resp)>;
        if constexpr (std::is_same_v<T, resp::error>) {
            format(err_, err_colors_, resp);
            had_error_ = true;
        } else if constexpr (std::is_same_v<T, resp::interrupted>) {
            format(err_, err_colors_, resp);
        } else if constexpr (std::is_same_v<T, resp::stopped>) {
            format(out_, colors_, resp);
            should_stop_ = true;
        } else {
            format(out_, colors_, resp);
        }
    }, r);
}

} // namespace mist::driver

#undef MIST_INLINE
