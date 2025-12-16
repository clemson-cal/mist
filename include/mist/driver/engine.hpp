#pragma once

#include <chrono>
#include <csignal>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include "command.hpp"
#include "physics_interface.hpp"
#include "response.hpp"
#include "state.hpp"
#include "../ascii_reader.hpp"
#include "../ascii_writer.hpp"
#include "../binary_reader.hpp"
#include "../binary_writer.hpp"
#include "../socket.hpp"

namespace mist::driver {

// =============================================================================
// Signal handling for Ctrl-C
// =============================================================================

namespace signal {

inline volatile std::sig_atomic_t interrupted = 0;

inline void handler(int) {
    interrupted = 1;
}

struct interrupt_guard_t {
    std::sig_atomic_t previous_state;
    void (*previous_handler)(int);

    interrupt_guard_t();
    ~interrupt_guard_t();
    auto is_interrupted() const -> bool;
    void clear();
};

} // namespace signal

// =============================================================================
// Utility
// =============================================================================

inline auto get_wall_time() -> double {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
// engine_t - the core state machine
// =============================================================================

class engine_t {
public:
    using emit_fn = std::function<void(const response_t&)>;

    engine_t(state_t& state, physics_interface_t& physics);
    ~engine_t();

    // Non-copyable, non-movable (owns socket)
    engine_t(const engine_t&) = delete;
    engine_t& operator=(const engine_t&) = delete;
    engine_t(engine_t&&) = delete;
    engine_t& operator=(engine_t&&) = delete;

    void execute(const command_t& cmd, emit_fn emit);
    void execute(const cmd::repeat_add& cmd, emit_fn emit);

    auto state() const -> const state_t& { return state_; }
    auto state() -> state_t& { return state_; }

    // Data socket for write commands
    auto data_socket_port() const -> int { return data_socket_.port(); }

    // Direct write methods - session handles I/O, engine handles data
    void write_physics(std::ostream& os, output_format fmt);
    void write_initial(std::ostream& os, output_format fmt);
    void write_driver(std::ostream& os, output_format fmt);
    void write_profiler(std::ostream& os, output_format fmt);
    void write_timeseries(std::ostream& os, output_format fmt);
    void write_checkpoint(std::ostream& os, output_format fmt);
    void write_products(std::ostream& os, output_format fmt);
    void write_iteration(std::ostream& os, output_format fmt);

    // Human-readable info methods (for show commands / REPL display)
    void write_iteration_info(std::ostream& os, const color::scheme_t& c);
    void write_profiler_info(std::ostream& os, const color::scheme_t& c);
    void write_timeseries_info(std::ostream& os, const color::scheme_t& c);

private:
    state_t& state_;
    physics_interface_t& physics_;
    double command_start_wall_time_;
    int command_start_iteration_;
    double last_dt_ = 0.0;
    socket_t data_socket_;

    auto make_iteration_info() const -> resp::iteration_info;
    auto time_to_next_task() const -> double;
    void write_to_socket(const std::function<void(std::ostream&)>& writer, emit_fn emit);

    void do_timestep(double dt_max);
    void execute_repeating_commands(emit_fn emit);
    void advance_to_target(const std::string& var, double target, emit_fn emit);

    void handle(const cmd::advance_by& c, emit_fn emit);
    void handle(const cmd::advance_to& c, emit_fn emit);
    void handle(const cmd::set_output& c, emit_fn emit);
    void handle(const cmd::set_physics& c, emit_fn emit);
    void handle(const cmd::set_initial& c, emit_fn emit);
    void handle(const cmd::set_exec& c, emit_fn emit);
    void handle(const cmd::select_timeseries& c, emit_fn emit);
    void handle(const cmd::select_products& c, emit_fn emit);
    void handle(const cmd::do_timeseries& c, emit_fn emit);
    void handle(const cmd::write_physics& c, emit_fn emit);
    void handle(const cmd::write_initial& c, emit_fn emit);
    void handle(const cmd::write_driver& c, emit_fn emit);
    void handle(const cmd::write_profiler& c, emit_fn emit);
    void handle(const cmd::write_timeseries& c, emit_fn emit);
    void handle(const cmd::write_checkpoint& c, emit_fn emit);
    void handle(const cmd::write_products& c, emit_fn emit);
    void handle(const cmd::write_iteration& c, emit_fn emit);
    void handle(const cmd::repeat_add& c, emit_fn emit);
    void handle(const cmd::clear_repeat& c, emit_fn emit);
    void handle(const cmd::init& c, emit_fn emit);
    void handle(const cmd::reset& c, emit_fn emit);
    void handle(const cmd::load& c, emit_fn emit);
    void handle(const cmd::show_state& c, emit_fn emit);
    void handle(const cmd::show_all& c, emit_fn emit);
    void handle(const cmd::show_physics& c, emit_fn emit);
    void handle(const cmd::show_initial& c, emit_fn emit);
    void handle(const cmd::show_iteration& c, emit_fn emit);
    void handle(const cmd::show_timeseries& c, emit_fn emit);
    void handle(const cmd::show_products& c, emit_fn emit);
    void handle(const cmd::show_profiler& c, emit_fn emit);
    void handle(const cmd::show_driver& c, emit_fn emit);
    void handle(const cmd::help& c, emit_fn emit);
    void handle(const cmd::help_schema& c, emit_fn emit);
    void handle(const cmd::stop& c, emit_fn emit);
};

} // namespace mist::driver

// =============================================================================
// Include implementations for header-only mode
// =============================================================================

#ifndef MIST_DRIVER_SEPARATE_COMPILATION
#include "engine.ipp"
#endif
