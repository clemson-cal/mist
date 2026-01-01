// engine.cpp - implementation of engine_t

#include <filesystem>
#include "mist/driver/engine.hpp"

namespace mist::driver {

namespace fs = std::filesystem;

// =============================================================================
// Help text
// =============================================================================

const char* help_text = R"(
  ---------------------------------------------------------------------------
  Stepping
  ---------------------------------------------------------------------------
    n++                            - Advance by 1 iteration
    n += 10                        - Advance by 10 iterations
    n -> 1000                      - Advance to iteration 1000
    t += 10.0                      - Advance time by exactly 10.0
    t -> 20.0                      - Advance time to exactly 20.0
    orbit += 3.0                   - Advance until orbit increases by 3.0
    orbit -> 60.0                  - Advance until orbit reaches 60.0

  ---------------------------------------------------------------------------
  Configuration
  ---------------------------------------------------------------------------
    set output=ascii               - Set output format (ascii|binary|hdf5)
    set physics key=val            - Set physics config parameter
    set initial key=val            - Set initial data parameter
    set exec key=val               - Set execution parameter (e.g. num_threads)
    select products [prod1 ...]    - Select products (no args = all)
    select timeseries [col1 ...]   - Select timeseries columns (no args = all)

  ---------------------------------------------------------------------------
  State management
  ---------------------------------------------------------------------------
    init                           - Generate initial state from config
    reset                          - Reset driver and clear physics state
    load <file>                    - Load checkpoint or config file

  ---------------------------------------------------------------------------
  Sampling
  ---------------------------------------------------------------------------
    do timeseries                  - Record timeseries sample

  ---------------------------------------------------------------------------
  File I/O
  ---------------------------------------------------------------------------
    write physics <file>           - Write physics config
    write initial <file>           - Write initial config
    write driver <file>            - Write driver state
    write profiler <file>          - Write profiler data
    write timeseries [file]        - Write timeseries
    write checkpoint [file]        - Write checkpoint
    write products [file]          - Write products
    write iteration [file]         - Write iteration info

  ---------------------------------------------------------------------------
  repeating commands
  ---------------------------------------------------------------------------
    repeat <interval> <unit> <cmd> - Execute command every interval
    repeat list                    - Show repeating commands
    clear repeat                   - Clear all repeating commands

  ---------------------------------------------------------------------------
  Information
  ---------------------------------------------------------------------------
    show                           - Show state summary
    show physics                   - Show physics configuration
    show initial                   - Show initial configuration
    show iteration                 - Show iteration info (n, t, dt, zps)
    show products                  - Show available and selected products
    show timeseries                - Show timeseries columns
    show driver                    - Show driver state
    show profiler                  - Show profiler
    show exec                      - Show execution context
    help                           - Show this help
    help schema                    - Show command/response schema
    stop | quit | q                - Exit simulation
)";

const char* schema_text = R"(
Commands:
  advance_by        { var: string, delta: double }
  advance_to        { var: string, target: double }
  set_output        { format: string }
  set_physics       { key: string, value: string }
  set_initial       { key: string, value: string }
  set_exec          { key: string, value: string }
  select_timeseries { cols: [string] }
  select_products   { prods: [string] }
  do_timeseries     { }
  write_physics     { dest: string }
  write_initial     { dest: string }
  write_driver      { dest: string }
  write_profiler    { dest: string }
  write_timeseries  { dest?: string }
  write_checkpoint  { dest?: string }
  write_products    { dest?: string }
  write_iteration   { dest?: string }
  repeat_add        { interval: double, unit: string, sub_command: command }
  clear_repeat      { }
  init              { }
  reset             { }
  load              { filename: string }
  show_*            { }
  help              { }
  stop              { }

Responses:
  ok                { message: string }
  error             { what: string }
  interrupted       { }
  stopped           { }
  state_info        { initialized: bool, zone_count: int, times: {string: double} }
  iteration_info    { n: int, times: {string: double}, dt: double, zps: double }
  timeseries_sample { string: double}
  physics_config    { text: string }
  initial_config    { text: string }
  driver_state      { text: string }
  help_text         { text: string }
  timeseries_info   { available: [string], selected: [string], counts: {string: int} }
  products_info     { available: [string], selected: [string] }
  profiler_info     { entries: [{name, count, time}], total_time: double }
  exec_info         { num_threads: int, mpi_rank: int, mpi_size: int }
  wrote_file        { filename: string, bytes: int }
)";

// =============================================================================
// Signal handling
// =============================================================================

signal::interrupt_guard_t::interrupt_guard_t() {
    previous_state = interrupted;
    interrupted = 0;
    previous_handler = std::signal(SIGINT, handler);
}

signal::interrupt_guard_t::~interrupt_guard_t() {
    std::signal(SIGINT, previous_handler);
    interrupted = previous_state;
}

auto signal::interrupt_guard_t::is_interrupted() const -> bool {
    return interrupted != 0;
}

void signal::interrupt_guard_t::clear() {
    interrupted = 0;
}

// =============================================================================
// engine_t implementation
// =============================================================================

engine_t::engine_t(state_t& state, physics_interface_t& phys, comm_t c)
    : state_ref(state)
    , physics(phys)
    , comm(std::move(c))
    , command_start_wall_time(get_wall_time())
    , command_start_iteration(state.iteration)
{
    data_socket.listen(0); // port 0 = any available port
    exec_context.comm = &comm;
    physics.set_exec_context(exec_context);
}

engine_t::~engine_t() = default;

void engine_t::broadcast_command(command_t& cmd) {
    if (comm.size() <= 1) {
        return;
    }

    auto buffer = std::vector<char>{};

    // Rank 0 serializes the command
    if (comm.rank() == 0) {
        auto oss = std::ostringstream{};
        auto sink = binary_sink{oss};
        write(sink, cmd);
        auto str = oss.str();
        buffer.assign(str.begin(), str.end());
    }

    // Broadcast size and data
    comm.broadcast(buffer);

    // Non-root ranks deserialize
    if (comm.rank() != 0) {
        auto iss = std::istringstream{std::string{buffer.begin(), buffer.end()}};
        auto source = binary_source{iss};
        read(source, cmd);
    }
}

void engine_t::log(const std::string& message) {
    if (log_stream) {
        *log_stream << message << "\n";
        log_stream->flush();
    }
}

void engine_t::set_log_prefix(const std::string& prefix) {
    auto filename = prefix + "_rank" + std::to_string(comm.rank()) + ".log";
    log_stream.emplace(filename);
}

void engine_t::run_as_follower() {
    while (true) {
        // Wait for broadcast command from rank 0
        auto cmd = command_t{};
        broadcast_command(cmd);

        // Execute locally without re-broadcasting, discard responses
        execute_local(cmd, [](const response_t&) {});

        // Exit on stop command
        if (std::holds_alternative<cmd::stop>(cmd)) {
            break;
        }
    }
}

void engine_t::execute_local(const command_t& cmd, emit_fn emit) {
    command_start_wall_time = get_wall_time();
    command_start_iteration = state_ref.iteration;

    // Log command execution
    log("execute command index " + std::to_string(cmd.index()));

    try {
        std::visit([this, &emit](const auto& c) { handle(c, emit); }, cmd);
        log("  completed");
    } catch (const std::exception& e) {
        log(std::string("  error: ") + e.what());
        emit(resp::error{e.what()});
    } catch (...) {
        log("  error: unknown");
        emit(resp::error{"unknown error"});
    }
}

void engine_t::execute(const command_t& cmd, emit_fn emit) {
    auto broadcast_cmd = cmd;

    // In distributed mode, broadcast command from rank 0
    broadcast_command(broadcast_cmd);

    // Execute locally
    execute_local(broadcast_cmd, emit);
}

void engine_t::execute(const cmd::repeat_add& cmd, emit_fn emit) {
    handle(cmd, emit);
}

auto engine_t::make_iteration_info() const -> resp::iteration_info {
    auto info = resp::iteration_info{};
    info.n = state_ref.iteration;

    for (const auto& name : physics.time_names()) {
        info.times[name] = physics.get_time(name);
    }

    info.dt = last_dt;

    auto wall_elapsed = get_wall_time() - command_start_wall_time;
    auto iter_elapsed = state_ref.iteration - command_start_iteration;

    if (iter_elapsed > 0 && wall_elapsed > 0) {
        info.zps = (iter_elapsed * physics.zone_count()) / wall_elapsed;
    } else {
        info.zps = last_zps;
    }

    return info;
}

void engine_t::do_timestep(double dt_max) {
    auto time_names = physics.time_names();
    auto t0 = physics.get_time(time_names[0]);
    physics.advance(dt_max);
    auto t1 = physics.get_time(time_names[0]);
    last_dt = t1 - t0;
    state_ref.iteration++;
}

auto engine_t::time_to_next_task() const -> double {
    auto dt_max = std::numeric_limits<double>::infinity();

    for (const auto& rc : state_ref.repeating_commands) {
        if (rc.unit == "n") continue;

        auto current = physics.get_time(rc.unit);
        auto time_until = rc.time_until_due(current);

        if (time_until > 0) {
            dt_max = std::min(dt_max, time_until);
        }
    }

    return dt_max;
}

void engine_t::execute_repeating_commands(emit_fn emit) {
    for (auto& rc : state_ref.repeating_commands) {
        auto current = (rc.unit == "n")
            ? static_cast<double>(state_ref.iteration)
            : physics.get_time(rc.unit);

        if (rc.time_until_due(current) <= 0) {
            std::visit([this, &emit](const auto& c) { handle(c, emit); }, rc.sub_command);
            rc.last_executed = current;
        }
    }
}

void engine_t::advance_to_target(const std::string& var, double target, emit_fn emit) {
    if (!physics.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }

    auto guard = signal::interrupt_guard_t{};

    // Fire any due tasks (including newly registered ones) before stepping
    execute_repeating_commands(emit);

    if (var == "n") {
        auto target_n = static_cast<int>(target);
        while (state_ref.iteration < target_n && !guard.is_interrupted()) {
            auto dt_max = time_to_next_task();
            do_timestep(dt_max);
            execute_repeating_commands(emit);
        }
    } else {
        auto eps = 1e-12 * std::abs(target);
        while (physics.get_time(var) < target - eps && !guard.is_interrupted()) {
            auto time_to_target = target - physics.get_time(var);
            auto dt_max = std::min(time_to_target, time_to_next_task());
            do_timestep(dt_max);
            execute_repeating_commands(emit);
        }
    }

    // Store zps from this advance command for later queries
    auto wall_elapsed = get_wall_time() - command_start_wall_time;
    auto iter_elapsed = state_ref.iteration - command_start_iteration;
    if (iter_elapsed > 0 && wall_elapsed > 0) {
        last_zps = (iter_elapsed * physics.zone_count()) / wall_elapsed;
    }

    if (guard.is_interrupted()) {
        emit(resp::interrupted{});
    } else {
        emit(resp::ok{"done"});
    }
}

void engine_t::write_to_socket(const std::function<void(std::ostream&)>& writer, emit_fn emit) {
    emit(resp::socket_listening{static_cast<int>(data_socket.port())});

    auto guard = signal::interrupt_guard_t{};
    auto client = data_socket.accept_interruptible([&guard] { return guard.is_interrupted(); });

    if (!client) {
        emit(resp::socket_cancelled{});
        return;
    }

    // Write data to client with size prefix
    auto oss = std::ostringstream{};
    writer(oss);
    auto data = oss.str();
    client->send_with_size(data.data(), data.size());
    emit(resp::socket_sent{data.size()});
}

// -----------------------------------------------------------------------------
// Format conversion helper
// -----------------------------------------------------------------------------

auto to_archive_format(output_format fmt) -> archive::format {
    return fmt == output_format::binary ? archive::format::binary : archive::format::ascii;
}

// -----------------------------------------------------------------------------
// Direct write methods
// -----------------------------------------------------------------------------

void engine_t::write_physics(std::ostream& os, output_format fmt) {
    physics.write_physics(os, fmt);
}

void engine_t::write_initial(std::ostream& os, output_format fmt) {
    physics.write_initial(os, fmt);
}

void engine_t::write_driver(std::ostream& os, output_format fmt) {
    archive::with_sink(os, to_archive_format(fmt), [&](auto& sink) { write(sink, "driver_state", state_ref); });
}

void engine_t::write_profiler(std::ostream& os, output_format fmt) {
    auto data = physics.profiler_data();
    archive::with_sink(os, to_archive_format(fmt), [&](auto& sink) { write(sink, "profiler", data); });
}

void engine_t::write_profiler_info(std::ostream& os, const color::scheme_t& c) {
    auto data = physics.profiler_data();
    auto total_time = 0.0;
    auto entries = std::vector<resp::profiler_entry>{};

    for (const auto& [name, entry] : data) {
        entries.push_back({name, entry.count, entry.time});
        total_time += entry.time;
    }

    format(os, c, resp::profiler_info{entries, total_time});
}

void engine_t::write_timeseries(std::ostream& os, output_format fmt) {
    archive::with_sink(os, to_archive_format(fmt), [&](auto& sink) { write(sink, "timeseries", state_ref.timeseries); });
}

void engine_t::write_timeseries_info(std::ostream& os, const color::scheme_t& c) {
    auto info = resp::timeseries_info{};

    // Get available columns from physics
    info.available = physics.timeseries_names();

    // Selected columns are the keys in the timeseries map
    for (const auto& [col, values] : state_ref.timeseries) {
        info.selected.push_back(col);
        info.counts[col] = values.size();
    }

    format(os, c, info);
}

void engine_t::write_checkpoint(std::ostream& os, output_format fmt) {
    archive::with_sink(os, to_archive_format(fmt), [&](auto& sink) { write(sink, "driver_state", state_ref); });
    physics.write_state(os, fmt);
}

void engine_t::write_products(std::ostream& os, output_format fmt) {
    physics.write_products(os, fmt, state_ref.selected_products);
}

void engine_t::write_iteration(std::ostream& os, output_format fmt) {
    auto info = make_iteration_info();
    archive::with_sink(os, to_archive_format(fmt), [&](auto& sink) { write(sink, "iteration", info); });
}

void engine_t::write_iteration_info(std::ostream& os, const color::scheme_t& c) {
    format(os, c, make_iteration_info());
}

// -----------------------------------------------------------------------------
// Command handlers
// -----------------------------------------------------------------------------

void engine_t::handle(const cmd::advance_by& c, emit_fn emit) {
    if (!physics.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }

    if (c.var == "n") {
        auto target = static_cast<double>(state_ref.iteration) + c.delta;
        advance_to_target("n", target, emit);
    } else {
        auto current = physics.get_time(c.var);
        advance_to_target(c.var, current + c.delta, emit);
    }
}

void engine_t::handle(const cmd::advance_to& c, emit_fn emit) {
    advance_to_target(c.var, c.target, emit);
}

void engine_t::handle(const cmd::set_output& c, emit_fn emit) {
    state_ref.format = from_string(std::type_identity<output_format>{}, c.format);
    emit(resp::ok{"output format set to " + c.format});
}

void engine_t::handle(const cmd::set_physics& c, emit_fn emit) {
    physics.set_physics(c.key, c.value);
    emit(resp::ok{"physics " + c.key + " set to " + c.value});
}

void engine_t::handle(const cmd::set_initial& c, emit_fn emit) {
    if (physics.has_state()) {
        emit(resp::error{"cannot modify initial config when state exists; use 'reset' first"});
        return;
    }
    physics.set_initial(c.key, c.value);
    emit(resp::ok{"initial " + c.key + " set to " + c.value});
}

void engine_t::handle(const cmd::set_exec& c, emit_fn emit) {
    if (c.key == "num_threads") {
        auto n = std::stoul(c.value);
        exec_context.set_num_threads(n);
        emit(resp::ok{"exec " + c.key + " set to " + c.value});
    } else {
        emit(resp::error{"unknown exec parameter: " + c.key});
    }
}

void engine_t::handle(const cmd::select_timeseries& c, emit_fn emit) {
    auto available = physics.timeseries_names();
    auto cols = c.cols.empty() ? available : c.cols;

    for (const auto& col : cols) {
        if (std::find(available.begin(), available.end(), col) == available.end()) {
            emit(resp::error{"timeseries column not found: " + col});
            return;
        }
    }

    state_ref.timeseries.clear();
    for (const auto& col : cols) {
        state_ref.timeseries[col] = {};
    }

    emit(resp::ok{"selected " + std::to_string(cols.size()) + " timeseries columns"});
}

void engine_t::handle(const cmd::select_products& c, emit_fn emit) {
    auto available = physics.product_names();
    auto prods = c.prods.empty() ? available : c.prods;

    for (const auto& prod : prods) {
        if (std::find(available.begin(), available.end(), prod) == available.end()) {
            emit(resp::error{"product not found: " + prod});
            return;
        }
    }

    state_ref.selected_products = prods;
    emit(resp::ok{"selected " + std::to_string(prods.size()) + " products"});
}

void engine_t::handle(const cmd::do_timeseries&, emit_fn emit) {
    if (!physics.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }
    if (state_ref.timeseries.empty()) {
        emit(resp::error{"no timeseries selected; use 'select timeseries'"});
        return;
    }

    auto sample = resp::timeseries_sample{};
    for (auto& [col, values] : state_ref.timeseries) {
        auto value = physics.get_timeseries(col);
        values.push_back(value);
        sample[col] = value;
    }
    emit(sample);
}

void engine_t::handle(const cmd::write_physics& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_physics(os, output_format::binary);
        }, emit);
        return;
    }

    auto filename = c.dest.value_or("physics.cfg");
    auto fmt = infer_format_from_filename(filename);
    auto file = std::ofstream{filename, std::ios::binary};
    if (!file) {
        emit(resp::error{"failed to open " + filename});
        return;
    }
    write_physics(file, fmt);
    emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
}

void engine_t::handle(const cmd::write_initial& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_initial(os, output_format::binary);
        }, emit);
        return;
    }

    auto filename = c.dest.value_or("initial.cfg");
    auto fmt = infer_format_from_filename(filename);
    auto file = std::ofstream{filename, std::ios::binary};
    if (!file) {
        emit(resp::error{"failed to open " + filename});
        return;
    }
    write_initial(file, fmt);
    emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
}

void engine_t::handle(const cmd::write_driver& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_driver(os, output_format::binary);
        }, emit);
        return;
    }

    auto filename = c.dest.value_or("driver.cfg");
    auto fmt = infer_format_from_filename(filename);
    auto file = std::ofstream{filename, std::ios::binary};
    if (!file) {
        emit(resp::error{"failed to open " + filename});
        return;
    }
    write_driver(file, fmt);
    emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
}

void engine_t::handle(const cmd::write_profiler& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_profiler(os, output_format::binary);
        }, emit);
        return;
    }

    auto filename = c.dest.value_or("profiler.dat");
    auto fmt = infer_format_from_filename(filename);
    auto file = std::ofstream{filename, std::ios::binary};
    if (!file) {
        emit(resp::error{"failed to open " + filename});
        return;
    }
    write_profiler(file, fmt);
    emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
}

void engine_t::handle(const cmd::write_timeseries& c, emit_fn emit) {
    if (state_ref.timeseries.empty()) {
        emit(resp::error{"no timeseries selected; use 'select timeseries'"});
        return;
    }

    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_timeseries(os, output_format::binary);
        }, emit);
        return;
    }

    auto filename = std::string{};
    auto fmt = output_format{};

    if (c.dest) {
        filename = *c.dest;
        fmt = infer_format_from_filename(filename);
    } else {
        fmt = state_ref.format;
        auto ext = (fmt == output_format::ascii) ? ".dat" : ".bin";
        auto oss = std::ostringstream{};
        oss << "timeseries." << std::setw(4) << std::setfill('0') << state_ref.timeseries_count << ext;
        filename = oss.str();
        state_ref.timeseries_count++;
    }

    auto file = std::ofstream{filename, std::ios::binary};
    if (!file) {
        emit(resp::error{"failed to open " + filename});
        return;
    }
    write_timeseries(file, fmt);
    emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
}

void engine_t::handle(const cmd::write_checkpoint& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_checkpoint(os, output_format::binary);
        }, emit);
        return;
    }

    auto fmt = output_format{};

    if (comm.size() > 1) {
        // Distributed checkpoint: create directory with header + patches
        auto dirname = std::string{};
        if (c.dest) {
            dirname = *c.dest;
            // Strip extension if present for directory name
            if (auto dot = dirname.rfind('.'); dot != std::string::npos) {
                dirname = dirname.substr(0, dot);
            }
            fmt = infer_format_from_filename(*c.dest);
        } else {
            fmt = state_ref.format;
            auto oss = std::ostringstream{};
            oss << "chkpt." << std::setw(4) << std::setfill('0') << state_ref.checkpoint_count;
            dirname = oss.str();
            state_ref.checkpoint_count++;
        }

        // Rank 0 creates directory and writes driver state
        auto dir = fs::path{dirname};
        if (comm.rank() == 0) {
            std::error_code ec;
            fs::create_directories(dir, ec);
            if (ec) {
                emit(resp::error{"failed to create directory " + dirname + ": " + ec.message()});
                comm.barrier();
                return;
            }

            auto driver_filename = dir / (fmt == output_format::ascii ? "driver.dat" : "driver.bin");
            auto driver_file = std::ofstream{driver_filename, std::ios::binary};
            if (!driver_file) {
                emit(resp::error{"failed to open driver state file"});
                comm.barrier();
                return;
            }
            archive::with_sink(driver_file, to_archive_format(fmt), [&](auto& sink) { write(sink, "driver_state", state_ref); });
        }
        comm.barrier();

        // All ranks write physics state (header + patches)
        physics.write_state(dir, fmt);
        comm.barrier();

        if (comm.rank() == 0) {
            emit(resp::wrote_file{dirname, 0});
        }
    } else {
        // Single-rank checkpoint: write single file
        auto filename = std::string{};
        if (c.dest) {
            filename = *c.dest;
            fmt = infer_format_from_filename(filename);
        } else {
            fmt = state_ref.format;
            auto ext = (fmt == output_format::ascii) ? ".dat" : ".bin";
            auto oss = std::ostringstream{};
            oss << "chkpt." << std::setw(4) << std::setfill('0') << state_ref.checkpoint_count << ext;
            filename = oss.str();
            state_ref.checkpoint_count++;
        }

        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_checkpoint(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    }
}

void engine_t::handle(const cmd::write_products& c, emit_fn emit) {
    if (!physics.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }
    if (state_ref.selected_products.empty()) {
        emit(resp::error{"no products selected; use 'select products'"});
        return;
    }

    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_products(os, output_format::binary);
        }, emit);
        return;
    }

    auto fmt = output_format{};

    if (comm.size() > 1) {
        // Distributed products: create directory with header + patches
        auto dirname = std::string{};
        if (c.dest) {
            dirname = *c.dest;
            if (auto dot = dirname.rfind('.'); dot != std::string::npos) {
                dirname = dirname.substr(0, dot);
            }
            fmt = infer_format_from_filename(*c.dest);
        } else {
            fmt = state_ref.format;
            auto oss = std::ostringstream{};
            oss << "prods." << std::setw(4) << std::setfill('0') << state_ref.products_count;
            dirname = oss.str();
            state_ref.products_count++;
        }

        auto dir = fs::path{dirname};
        if (comm.rank() == 0) {
            std::error_code ec;
            fs::create_directories(dir, ec);
            if (ec) {
                emit(resp::error{"failed to create directory " + dirname + ": " + ec.message()});
                comm.barrier();
                return;
            }
        }
        comm.barrier();

        physics.write_products(dir, fmt, state_ref.selected_products);
        comm.barrier();

        if (comm.rank() == 0) {
            emit(resp::wrote_file{dirname, 0});
        }
    } else {
        // Single-rank products: write single file
        auto filename = std::string{};
        if (c.dest) {
            filename = *c.dest;
            fmt = infer_format_from_filename(filename);
        } else {
            fmt = state_ref.format;
            auto ext = (fmt == output_format::ascii) ? ".dat" : ".bin";
            auto oss = std::ostringstream{};
            oss << "prods." << std::setw(4) << std::setfill('0') << state_ref.products_count << ext;
            filename = oss.str();
            state_ref.products_count++;
        }

        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_products(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    }
}

void engine_t::handle(const cmd::write_iteration& c, emit_fn emit) {
    if (!physics.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }

    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_iteration(os, output_format::binary);
        }, emit);
        return;
    }

    if (c.dest) {
        auto fmt = infer_format_from_filename(*c.dest);
        auto file = std::ofstream{*c.dest, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + *c.dest});
            return;
        }
        write_iteration(file, fmt);
        emit(resp::wrote_file{*c.dest, static_cast<std::size_t>(file.tellp())});
    } else {
        emit(make_iteration_info());
    }
}

void engine_t::handle(const cmd::repeat_add& c, emit_fn emit) {
    if (!is_repeatable(c.sub_command)) {
        emit(resp::error{"command is not repeatable"});
        return;
    }

    if (c.unit != "n") {
        auto time_names = physics.time_names();
        if (std::find(time_names.begin(), time_names.end(), c.unit) == time_names.end()) {
            emit(resp::error{"unknown time variable: " + c.unit});
            return;
        }
    }

    auto rc = repeating_command_t{};
    rc.interval = c.interval;
    rc.unit = c.unit;
    rc.sub_command = c.sub_command;
    state_ref.repeating_commands.push_back(rc);

    emit(resp::ok{"added repeating command"});
}

void engine_t::handle(const cmd::clear_repeat&, emit_fn emit) {
    state_ref.repeating_commands.clear();
    emit(resp::ok{"cleared all repeating commands"});
}

void engine_t::handle(const cmd::init&, emit_fn emit) {
    if (physics.has_state()) {
        emit(resp::error{"state already initialized; use 'reset' first"});
        return;
    }
    physics.init();
    emit(resp::ok{"initialized physics state"});
}

void engine_t::handle(const cmd::reset&, emit_fn emit) {
    physics.reset();
    state_ref.iteration = 0;
    state_ref.checkpoint_count = 0;
    state_ref.products_count = 0;
    state_ref.timeseries_count = 0;
    state_ref.timeseries.clear();
    emit(resp::ok{"reset driver and physics state"});
}

void engine_t::handle(const cmd::load& c, emit_fn emit) {
    // Determine the path to load - may be file or directory
    auto path = c.filename;

    // If path doesn't exist but stripping extension gives a directory, use that
    // This handles: user says "load chkpt.0000.dat" but we wrote "chkpt.0000/"
    if (!fs::exists(path)) {
        if (auto dot = path.rfind('.'); dot != std::string::npos) {
            auto dirname = path.substr(0, dot);
            if (fs::is_directory(dirname)) {
                path = dirname;
            }
        }
    }

    // Check if path is a directory (distributed checkpoint)
    if (fs::is_directory(path)) {
        // Distributed checkpoint: header + patches structure
        auto fmt = output_format{};

        // Detect format from driver file
        auto dir = fs::path{path};
        auto driver_bin = dir / "driver.bin";
        auto driver_dat = dir / "driver.dat";
        fs::path driver_file;
        if (fs::exists(driver_bin)) {
            driver_file = driver_bin;
            fmt = output_format::binary;
        } else if (fs::exists(driver_dat)) {
            driver_file = driver_dat;
            fmt = output_format::ascii;
        } else {
            comm.barrier();
            if (comm.rank() == 0) {
                emit(resp::error{"driver file not found in " + path});
            }
            return;
        }

        // All ranks read driver state
        auto success = false;
        {
            auto file = std::ifstream{driver_file, std::ios::binary};
            if (!file) {
                comm.barrier();
                if (comm.rank() == 0) {
                    emit(resp::error{"failed to open " + driver_file.string()});
                }
                return;
            }
            success = archive::with_source(file, to_archive_format(fmt), [&](auto& source) {
                return read(source, "driver_state", state_ref);
            });
        }

        // Load physics state (header + patches by affinity)
        if (success) {
            success = physics.load_state(fs::path{path}, fmt);
        }

        // Check if all ranks succeeded
        auto all_success = comm.combine(success ? 1 : 0, [](int a, int b) { return a * b; }) != 0;
        comm.barrier();

        if (comm.rank() == 0) {
            if (all_success) {
                emit(resp::ok{"loaded distributed checkpoint from " + path});
            } else {
                emit(resp::error{"failed to load distributed checkpoint from " + path});
            }
        }
        return;
    }

    // Single file: try various formats
    auto fmt = infer_format_from_filename(c.filename);

    auto file = std::ifstream{c.filename, std::ios::binary};
    if (!file) {
        emit(resp::error{"failed to open " + c.filename});
        return;
    }

    auto loaded = archive::with_source(file, to_archive_format(fmt), [&](auto& source) {
        return read(source, "driver_state", state_ref);
    });
    if (loaded && physics.load_state(file, fmt)) {
        emit(resp::ok{"loaded checkpoint from " + c.filename});
        return;
    }

    file.clear();
    file.seekg(0);
    if (physics.load_physics(file, fmt)) {
        physics.reset();
        state_ref.iteration = 0;
        emit(resp::ok{"loaded physics config from " + c.filename + "; use 'init' to generate state"});
        return;
    }

    file.clear();
    file.seekg(0);
    if (physics.load_initial(file, fmt)) {
        emit(resp::ok{"loaded initial config from " + c.filename});
        return;
    }

    emit(resp::error{"could not load checkpoint, physics, or initial from " + c.filename});
}

void engine_t::handle(const cmd::show_state&, emit_fn emit) {
    auto info = resp::state_info{};
    info.initialized = physics.has_state();
    if (info.initialized) {
        info.zone_count = physics.zone_count();
        for (const auto& name : physics.time_names()) {
            info.times[name] = physics.get_time(name);
        }
    }
    emit(info);
}

void engine_t::handle(const cmd::show_all&, emit_fn emit) {
    handle(cmd::show_exec{}, emit);
    handle(cmd::show_products{}, emit);
    handle(cmd::show_timeseries{}, emit);
    handle(cmd::show_physics{}, emit);
    handle(cmd::show_initial{}, emit);
    handle(cmd::show_driver{}, emit);
}

void engine_t::handle(const cmd::show_physics&, emit_fn emit) {
    auto oss = std::ostringstream{};
    physics.write_physics(oss, output_format::ascii);
    emit(resp::physics_config{oss.str()});
}

void engine_t::handle(const cmd::show_initial&, emit_fn emit) {
    auto oss = std::ostringstream{};
    physics.write_initial(oss, output_format::ascii);
    emit(resp::initial_config{oss.str()});
}

void engine_t::handle(const cmd::show_iteration&, emit_fn emit) {
    if (!physics.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }
    emit(make_iteration_info());
}

void engine_t::handle(const cmd::show_timeseries&, emit_fn emit) {
    auto info = resp::timeseries_info{};
    info.available = physics.timeseries_names();
    for (const auto& [col, values] : state_ref.timeseries) {
        info.selected.push_back(col);
        info.counts[col] = values.size();
    }
    emit(info);
}

void engine_t::handle(const cmd::show_products&, emit_fn emit) {
    auto info = resp::products_info{};
    info.available = physics.product_names();
    info.selected = state_ref.selected_products;
    emit(info);
}

void engine_t::handle(const cmd::show_profiler&, emit_fn emit) {
    auto data = physics.profiler_data();
    auto info = resp::profiler_info{};

    for (const auto& [name, entry] : data) {
        info.entries.push_back({name, entry.count, entry.time});
        info.total_time += entry.time;
    }

    std::sort(info.entries.begin(), info.entries.end(),
        [](const auto& a, const auto& b) { return a.time > b.time; });

    emit(info);
}

void engine_t::handle(const cmd::show_driver&, emit_fn emit) {
    auto oss = std::ostringstream{};
    auto sink = ascii_sink{oss};
    write(sink, "driver_state", state_ref);
    emit(resp::driver_state{oss.str()});
}

void engine_t::handle(const cmd::show_exec&, emit_fn emit) {
    emit(resp::exec_info{
        static_cast<int>(exec_context.num_threads()),
        exec_context.mpi_rank(),
        exec_context.mpi_size()
    });
}

void engine_t::handle(const cmd::help&, emit_fn emit) {
    emit(resp::help_text{help_text});
}

void engine_t::handle(const cmd::help_schema&, emit_fn emit) {
    emit(resp::help_text{schema_text});
}

void engine_t::handle(const cmd::stop&, emit_fn emit) {
    emit(resp::stopped{});
}

} // namespace mist::driver
