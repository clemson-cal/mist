// distributed_session.cpp - implementation of distributed_session_t

#include "mist/driver/distributed_session.hpp"
#include <sstream>
#include <unistd.h>
#include <readline/readline.h>
#include <readline/history.h>

namespace mist::driver {

// =============================================================================
// distributed_session_t implementation
// =============================================================================

distributed_session_t::distributed_session_t(engine_t& engine, communicator_t& comm,
                                             std::ostream& out, std::ostream& err)
    : engine_(engine)
    , comm_(comm)
    , out_(out)
    , err_(err)
    , colors_(color::for_stream(out))
    , err_colors_(color::for_stream(err))
{
}

int distributed_session_t::run() {
    auto is_tty = comm_.is_root() && isatty(STDIN_FILENO);

    // Warn about interactive mode with MPI (stdin forwarding issues)
    if (is_tty && comm_.size() > 1) {
        err_ << err_colors_.warning << "warning: " << err_colors_.reset
             << "interactive mode with MPI may hang; use piped input or --socket\n";
    }

    while (true) {
        auto input = std::string{};

        // Root reads command
        if (comm_.is_root()) {
            std::cout.flush();  // Ensure prompt is displayed
            auto* line = readline(is_tty ? "> " : "");
            if (!line) {
                input = "stop";  // EOF -> stop all ranks
            } else {
                input = line;
                std::free(line);

                if (!is_tty && !input.empty() && input[0] != '#') {
                    out_ << "> " << input << "\n";
                }

                if (!input.empty()) {
                    add_history(input.c_str());
                }
            }
        }

        // Broadcast command string to all ranks
        comm_.broadcast_string(input);

        // Skip empty lines and comments
        if (input.empty() || input[0] == '#') {
            continue;
        }

        // Parse and execute
        auto parsed = parse_command(input);

        if (!parsed.cmd && !parsed.repeat) {
            if (comm_.is_root()) {
                err_ << err_colors_.error << "error: " << err_colors_.reset
                     << parsed.error << "\n";
            }
            if (!is_tty) {
                break;
            }
            continue;
        }

        // All ranks execute
        had_error_ = false;
        should_stop_ = false;

        auto emit = [this](const response_t& r) {
            // Only root formats output
            if (comm_.is_root()) {
                format_response(r);
            }
        };

        if (parsed.repeat) {
            engine_.execute(*parsed.repeat, emit);
        } else {
            engine_.execute(*parsed.cmd, emit);
        }

        // Synchronize after command
        comm_.barrier();

        if (had_error_ && !is_tty) {
            break;
        }
        if (should_stop_) {
            break;
        }
    }

    return 0;
}

void distributed_session_t::format_response(const response_t& r) {
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
