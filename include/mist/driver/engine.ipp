// engine.ipp - implementation file for engine_t
// Include via engine.hpp (header-only) or compile engine.cpp (separate compilation)

#ifdef MIST_DRIVER_SEPARATE_COMPILATION
#define MIST_INLINE
#else
#define MIST_INLINE inline
#endif

namespace mist::driver {

// =============================================================================
// Help text
// =============================================================================

MIST_INLINE const char* help_text = R"(
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
    help                           - Show this help
    help schema                    - Show command/response schema
    stop | quit | q                - Exit simulation
)";

MIST_INLINE const char* schema_text = R"(
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
  timeseries_sample { values: {string: double} }
  physics_config    { text: string }
  initial_config    { text: string }
  driver_state      { text: string }
  help_text         { text: string }
  timeseries_info   { available: [string], selected: [string], counts: {string: int} }
  products_info     { available: [string], selected: [string] }
  profiler_info     { entries: [{name, count, time}], total_time: double }
  wrote_file        { filename: string, bytes: int }
)";

// =============================================================================
// Signal handling
// =============================================================================

MIST_INLINE signal::interrupt_guard_t::interrupt_guard_t() {
    previous_state = interrupted;
    interrupted = 0;
    previous_handler = std::signal(SIGINT, handler);
}

MIST_INLINE signal::interrupt_guard_t::~interrupt_guard_t() {
    std::signal(SIGINT, previous_handler);
    interrupted = previous_state;
}

MIST_INLINE auto signal::interrupt_guard_t::is_interrupted() const -> bool {
    return interrupted != 0;
}

MIST_INLINE void signal::interrupt_guard_t::clear() {
    interrupted = 0;
}

// =============================================================================
// engine_t implementation
// =============================================================================

MIST_INLINE engine_t::engine_t(state_t& state, physics_interface_t& physics)
    : state_(state)
    , physics_(physics)
    , command_start_wall_time_(get_wall_time())
    , command_start_iteration_(state.iteration)
{
    data_socket_.listen(0); // port 0 = any available port
}

MIST_INLINE engine_t::~engine_t() = default;

MIST_INLINE void engine_t::execute(const command_t& cmd, emit_fn emit) {
    command_start_wall_time_ = get_wall_time();
    command_start_iteration_ = state_.iteration;
    std::visit([this, &emit](const auto& c) { handle(c, emit); }, cmd);
}

MIST_INLINE void engine_t::execute(const cmd::repeat_add& cmd, emit_fn emit) {
    handle(cmd, emit);
}

MIST_INLINE auto engine_t::make_iteration_info() const -> resp::iteration_info {
    auto info = resp::iteration_info{};
    info.n = state_.iteration;

    for (const auto& name : physics_.time_names()) {
        info.times[name] = physics_.get_time(name);
    }

    info.dt = last_dt_;

    auto wall_elapsed = get_wall_time() - command_start_wall_time_;
    auto iter_elapsed = state_.iteration - command_start_iteration_;
    info.zps = (wall_elapsed > 0)
        ? (iter_elapsed * physics_.zone_count()) / wall_elapsed
        : 0.0;

    return info;
}

MIST_INLINE void engine_t::do_timestep(double dt_max) {
    auto time_names = physics_.time_names();
    auto t0 = physics_.get_time(time_names[0]);
    physics_.advance(dt_max);
    auto t1 = physics_.get_time(time_names[0]);
    last_dt_ = t1 - t0;
    state_.iteration++;
}

MIST_INLINE auto engine_t::time_to_next_task() const -> double {
    auto dt_max = std::numeric_limits<double>::infinity();

    for (const auto& rc : state_.repeating_commands) {
        if (rc.unit == "n") continue;

        auto current = physics_.get_time(rc.unit);
        auto time_until = rc.time_until_due(current);

        if (time_until > 0) {
            dt_max = std::min(dt_max, time_until);
        }
    }

    return dt_max;
}

MIST_INLINE void engine_t::execute_repeating_commands(emit_fn emit) {
    for (auto& rc : state_.repeating_commands) {
        auto current = (rc.unit == "n")
            ? static_cast<double>(state_.iteration)
            : physics_.get_time(rc.unit);

        if (rc.time_until_due(current) <= 0) {
            std::visit([this, &emit](const auto& c) { handle(c, emit); }, rc.sub_command);
            rc.last_executed = current;
        }
    }
}

MIST_INLINE void engine_t::advance_to_target(const std::string& var, double target, emit_fn emit) {
    if (!physics_.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }

    auto guard = signal::interrupt_guard_t{};

    // Fire any due tasks (including newly registered ones) before stepping
    execute_repeating_commands(emit);

    if (var == "n") {
        auto target_n = static_cast<int>(target);
        while (state_.iteration < target_n && !guard.is_interrupted()) {
            auto dt_max = time_to_next_task();
            do_timestep(dt_max);
            execute_repeating_commands(emit);
        }
    } else {
        auto eps = 1e-12 * std::abs(target);
        while (physics_.get_time(var) < target - eps && !guard.is_interrupted()) {
            auto time_to_target = target - physics_.get_time(var);
            auto dt_max = std::min(time_to_target, time_to_next_task());
            do_timestep(dt_max);
            execute_repeating_commands(emit);
        }
    }

    if (guard.is_interrupted()) {
        emit(resp::interrupted{});
    } else {
        emit(resp::ok{"done"});
    }
}

MIST_INLINE void engine_t::write_to_socket(const std::function<void(std::ostream&)>& writer, emit_fn emit) {
    emit(resp::socket_listening{static_cast<int>(data_socket_.port())});

    auto guard = signal::interrupt_guard_t{};
    auto client = data_socket_.accept_interruptible([&guard] { return guard.is_interrupted(); });

    if (!client) {
        emit(resp::socket_cancelled{});
        return;
    }

    try {
        // Write data to client with size prefix
        auto oss = std::ostringstream{};
        writer(oss);
        auto data = oss.str();
        client->send_with_size(data.data(), data.size());
        emit(resp::socket_sent{data.size()});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

// -----------------------------------------------------------------------------
// Direct write methods
// -----------------------------------------------------------------------------

MIST_INLINE void engine_t::write_physics(std::ostream& os, output_format fmt) {
    physics_.write_physics(os, fmt);
}

MIST_INLINE void engine_t::write_initial(std::ostream& os, output_format fmt) {
    physics_.write_initial(os, fmt);
}

MIST_INLINE void engine_t::write_driver(std::ostream& os, output_format fmt) {
    if (fmt == output_format::binary) {
        auto writer = binary_writer{os};
        serialize(writer, "driver_state", state_);
    } else {
        auto writer = ascii_writer{os};
        serialize(writer, "driver_state", state_);
    }
}

MIST_INLINE void engine_t::write_profiler(std::ostream& os, output_format fmt) {
    auto data = physics_.profiler_data();
    if (fmt == output_format::binary) {
        auto writer = binary_writer{os};
        serialize(writer, "profiler", data);
    } else {
        auto writer = ascii_writer{os};
        serialize(writer, "profiler", data);
    }
}

MIST_INLINE void engine_t::write_profiler_info(std::ostream& os, const color::scheme_t& c) {
    auto data = physics_.profiler_data();
    auto total_time = 0.0;
    auto entries = std::vector<resp::profiler_entry>{};

    for (const auto& [name, entry] : data) {
        entries.push_back({name, entry.count, entry.time});
        total_time += entry.time;
    }

    format(os, c, resp::profiler_info{entries, total_time});
}

MIST_INLINE void engine_t::write_timeseries(std::ostream& os, output_format fmt) {
    if (fmt == output_format::binary) {
        auto writer = binary_writer{os};
        serialize(writer, "timeseries", state_.timeseries);
    } else {
        auto writer = ascii_writer{os};
        serialize(writer, "timeseries", state_.timeseries);
    }
}

MIST_INLINE void engine_t::write_timeseries_info(std::ostream& os, const color::scheme_t& c) {
    auto info = resp::timeseries_info{};

    // Get available columns from physics
    info.available = physics_.timeseries_names();

    // Selected columns are the keys in the timeseries map
    for (const auto& [col, values] : state_.timeseries) {
        info.selected.push_back(col);
        info.counts[col] = values.size();
    }

    format(os, c, info);
}

MIST_INLINE void engine_t::write_checkpoint(std::ostream& os, output_format fmt) {
    if (fmt == output_format::binary) {
        auto writer = binary_writer{os};
        serialize(writer, "driver_state", state_);
    } else {
        auto writer = ascii_writer{os};
        serialize(writer, "driver_state", state_);
    }
    physics_.write_state(os, fmt);
}

MIST_INLINE void engine_t::write_products(std::ostream& os, output_format fmt) {
    physics_.write_products(os, fmt, state_.selected_products);
}

MIST_INLINE void engine_t::write_iteration(std::ostream& os, output_format fmt) {
    auto info = make_iteration_info();
    if (fmt == output_format::binary) {
        auto writer = binary_writer{os};
        serialize(writer, "iteration", info);
    } else {
        auto writer = ascii_writer{os};
        serialize(writer, "iteration", info);
    }
}

MIST_INLINE void engine_t::write_iteration_info(std::ostream& os, const color::scheme_t& c) {
    format(os, c, make_iteration_info());
}

// -----------------------------------------------------------------------------
// Command handlers
// -----------------------------------------------------------------------------

MIST_INLINE void engine_t::handle(const cmd::advance_by& c, emit_fn emit) {
    if (!physics_.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }

    if (c.var == "n") {
        auto target = static_cast<double>(state_.iteration) + c.delta;
        advance_to_target("n", target, emit);
    } else {
        auto current = physics_.get_time(c.var);
        advance_to_target(c.var, current + c.delta, emit);
    }
}

MIST_INLINE void engine_t::handle(const cmd::advance_to& c, emit_fn emit) {
    advance_to_target(c.var, c.target, emit);
}

MIST_INLINE void engine_t::handle(const cmd::set_output& c, emit_fn emit) {
    try {
        state_.format = from_string(std::type_identity<output_format>{}, c.format);
        emit(resp::ok{"output format set to " + c.format});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::set_physics& c, emit_fn emit) {
    try {
        physics_.set_physics(c.key, c.value);
        emit(resp::ok{"physics " + c.key + " set to " + c.value});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::set_initial& c, emit_fn emit) {
    if (physics_.has_state()) {
        emit(resp::error{"cannot modify initial config when state exists; use 'reset' first"});
        return;
    }
    try {
        physics_.set_initial(c.key, c.value);
        emit(resp::ok{"initial " + c.key + " set to " + c.value});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::set_exec& c, emit_fn emit) {
    try {
        physics_.set_exec(c.key, c.value);
        emit(resp::ok{"exec " + c.key + " set to " + c.value});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::select_timeseries& c, emit_fn emit) {
    auto available = physics_.timeseries_names();
    auto cols = c.cols.empty() ? available : c.cols;

    for (const auto& col : cols) {
        if (std::find(available.begin(), available.end(), col) == available.end()) {
            emit(resp::error{"timeseries column not found: " + col});
            return;
        }
    }

    state_.timeseries.clear();
    for (const auto& col : cols) {
        state_.timeseries[col] = {};
    }

    emit(resp::ok{"selected " + std::to_string(cols.size()) + " timeseries columns"});
}

MIST_INLINE void engine_t::handle(const cmd::select_products& c, emit_fn emit) {
    auto available = physics_.product_names();
    auto prods = c.prods.empty() ? available : c.prods;

    for (const auto& prod : prods) {
        if (std::find(available.begin(), available.end(), prod) == available.end()) {
            emit(resp::error{"product not found: " + prod});
            return;
        }
    }

    state_.selected_products = prods;
    emit(resp::ok{"selected " + std::to_string(prods.size()) + " products"});
}

MIST_INLINE void engine_t::handle(const cmd::do_timeseries&, emit_fn emit) {
    if (!physics_.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }
    if (state_.timeseries.empty()) {
        emit(resp::error{"no timeseries selected; use 'select timeseries'"});
        return;
    }

    auto sample = resp::timeseries_sample{};
    for (auto& [col, values] : state_.timeseries) {
        auto value = physics_.get_timeseries(col);
        values.push_back(value);
        sample.values[col] = value;
    }
    emit(sample);
}

MIST_INLINE void engine_t::handle(const cmd::write_physics& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_physics(os, output_format::binary);
        }, emit);
        return;
    }

    try {
        auto filename = c.dest.value_or("physics.cfg");
        auto fmt = infer_format_from_filename(filename);
        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_physics(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::write_initial& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_initial(os, output_format::binary);
        }, emit);
        return;
    }

    try {
        auto filename = c.dest.value_or("initial.cfg");
        auto fmt = infer_format_from_filename(filename);
        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_initial(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::write_driver& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_driver(os, output_format::binary);
        }, emit);
        return;
    }

    try {
        auto filename = c.dest.value_or("driver.cfg");
        auto fmt = infer_format_from_filename(filename);
        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_driver(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::write_profiler& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_profiler(os, output_format::binary);
        }, emit);
        return;
    }

    try {
        auto filename = c.dest.value_or("profiler.dat");
        auto fmt = infer_format_from_filename(filename);
        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_profiler(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::write_timeseries& c, emit_fn emit) {
    if (state_.timeseries.empty()) {
        emit(resp::error{"no timeseries selected; use 'select timeseries'"});
        return;
    }

    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_timeseries(os, output_format::binary);
        }, emit);
        return;
    }

    try {
        auto filename = std::string{};
        auto fmt = output_format{};

        if (c.dest) {
            filename = *c.dest;
            fmt = infer_format_from_filename(filename);
        } else {
            fmt = state_.format;
            auto ext = (fmt == output_format::ascii) ? ".dat" : ".bin";
            auto oss = std::ostringstream{};
            oss << "timeseries." << std::setw(4) << std::setfill('0') << state_.timeseries_count << ext;
            filename = oss.str();
            state_.timeseries_count++;
        }

        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_timeseries(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::write_checkpoint& c, emit_fn emit) {
    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_checkpoint(os, output_format::binary);
        }, emit);
        return;
    }

    try {
        auto filename = std::string{};
        auto fmt = output_format{};

        if (c.dest) {
            filename = *c.dest;
            fmt = infer_format_from_filename(filename);
        } else {
            fmt = state_.format;
            auto ext = (fmt == output_format::ascii) ? ".dat" : ".bin";
            auto oss = std::ostringstream{};
            oss << "chkpt." << std::setw(4) << std::setfill('0') << state_.checkpoint_count << ext;
            filename = oss.str();
            state_.checkpoint_count++;
        }

        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_checkpoint(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::write_products& c, emit_fn emit) {
    if (!physics_.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }
    if (state_.selected_products.empty()) {
        emit(resp::error{"no products selected; use 'select products'"});
        return;
    }

    if (c.dest && *c.dest == "socket") {
        write_to_socket([this](std::ostream& os) {
            write_products(os, output_format::binary);
        }, emit);
        return;
    }

    try {
        auto filename = std::string{};
        auto fmt = output_format{};

        if (c.dest) {
            filename = *c.dest;
            fmt = infer_format_from_filename(filename);
        } else {
            fmt = state_.format;
            auto ext = (fmt == output_format::ascii) ? ".dat" : ".bin";
            auto oss = std::ostringstream{};
            oss << "prods." << std::setw(4) << std::setfill('0') << state_.products_count << ext;
            filename = oss.str();
            state_.products_count++;
        }

        auto file = std::ofstream{filename, std::ios::binary};
        if (!file) {
            emit(resp::error{"failed to open " + filename});
            return;
        }
        write_products(file, fmt);
        emit(resp::wrote_file{filename, static_cast<std::size_t>(file.tellp())});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::write_iteration& c, emit_fn emit) {
    if (!physics_.has_state()) {
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
        try {
            auto fmt = infer_format_from_filename(*c.dest);
            auto file = std::ofstream{*c.dest, std::ios::binary};
            if (!file) {
                emit(resp::error{"failed to open " + *c.dest});
                return;
            }
            write_iteration(file, fmt);
            emit(resp::wrote_file{*c.dest, static_cast<std::size_t>(file.tellp())});
        } catch (const std::exception& e) {
            emit(resp::error{e.what()});
        }
    } else {
        emit(make_iteration_info());
    }
}

MIST_INLINE void engine_t::handle(const cmd::repeat_add& c, emit_fn emit) {
    if (!is_repeatable(c.sub_command)) {
        emit(resp::error{"command is not repeatable"});
        return;
    }

    if (c.unit != "n") {
        auto time_names = physics_.time_names();
        if (std::find(time_names.begin(), time_names.end(), c.unit) == time_names.end()) {
            emit(resp::error{"unknown time variable: " + c.unit});
            return;
        }
    }

    auto rc = repeating_command_t{};
    rc.interval = c.interval;
    rc.unit = c.unit;
    rc.sub_command = c.sub_command;
    state_.repeating_commands.push_back(rc);

    emit(resp::ok{"added repeating command"});
}

MIST_INLINE void engine_t::handle(const cmd::clear_repeat&, emit_fn emit) {
    state_.repeating_commands.clear();
    emit(resp::ok{"cleared all repeating commands"});
}

MIST_INLINE void engine_t::handle(const cmd::init&, emit_fn emit) {
    if (physics_.has_state()) {
        emit(resp::error{"state already initialized; use 'reset' first"});
        return;
    }
    try {
        physics_.init();
        emit(resp::ok{"initialized physics state"});
    } catch (const std::exception& e) {
        emit(resp::error{e.what()});
    }
}

MIST_INLINE void engine_t::handle(const cmd::reset&, emit_fn emit) {
    physics_.reset();
    state_.iteration = 0;
    state_.checkpoint_count = 0;
    state_.products_count = 0;
    state_.timeseries_count = 0;
    state_.timeseries.clear();
    emit(resp::ok{"reset driver and physics state"});
}

MIST_INLINE void engine_t::handle(const cmd::load& c, emit_fn emit) {
    auto fmt = infer_format_from_filename(c.filename);

    auto file = std::ifstream{c.filename, std::ios::binary};
    if (!file) {
        emit(resp::error{"failed to open " + c.filename});
        return;
    }

    if (fmt == output_format::ascii) {
        auto reader = ascii_reader{file};
        if (deserialize(reader, "driver_state", state_) && physics_.load_state(file, fmt)) {
            emit(resp::ok{"loaded checkpoint from " + c.filename});
            return;
        }
    } else {
        auto reader = binary_reader{file};
        if (deserialize(reader, "driver_state", state_) && physics_.load_state(file, fmt)) {
            emit(resp::ok{"loaded checkpoint from " + c.filename});
            return;
        }
    }

    file.clear();
    file.seekg(0);
    if (physics_.load_physics(file, fmt)) {
        physics_.reset();
        state_.iteration = 0;
        emit(resp::ok{"loaded physics config from " + c.filename + "; use 'init' to generate state"});
        return;
    }

    file.clear();
    file.seekg(0);
    if (physics_.load_initial(file, fmt)) {
        emit(resp::ok{"loaded initial config from " + c.filename});
        return;
    }

    emit(resp::error{"could not load checkpoint, physics, or initial from " + c.filename});
}

MIST_INLINE void engine_t::handle(const cmd::show_state&, emit_fn emit) {
    auto info = resp::state_info{};
    info.initialized = physics_.has_state();
    if (info.initialized) {
        info.zone_count = physics_.zone_count();
        for (const auto& name : physics_.time_names()) {
            info.times[name] = physics_.get_time(name);
        }
    }
    emit(info);
}

MIST_INLINE void engine_t::handle(const cmd::show_all&, emit_fn emit) {
    handle(cmd::show_state{}, emit);
    handle(cmd::show_physics{}, emit);
    handle(cmd::show_initial{}, emit);
    handle(cmd::show_driver{}, emit);
    handle(cmd::show_products{}, emit);
    handle(cmd::show_timeseries{}, emit);
    handle(cmd::show_profiler{}, emit);
}

MIST_INLINE void engine_t::handle(const cmd::show_physics&, emit_fn emit) {
    auto oss = std::ostringstream{};
    physics_.write_physics(oss, output_format::ascii);
    emit(resp::physics_config{oss.str()});
}

MIST_INLINE void engine_t::handle(const cmd::show_initial&, emit_fn emit) {
    auto oss = std::ostringstream{};
    physics_.write_initial(oss, output_format::ascii);
    emit(resp::initial_config{oss.str()});
}

MIST_INLINE void engine_t::handle(const cmd::show_iteration&, emit_fn emit) {
    if (!physics_.has_state()) {
        emit(resp::error{"physics state not initialized; use 'init' first"});
        return;
    }
    emit(make_iteration_info());
}

MIST_INLINE void engine_t::handle(const cmd::show_timeseries&, emit_fn emit) {
    auto info = resp::timeseries_info{};
    info.available = physics_.timeseries_names();
    for (const auto& [col, values] : state_.timeseries) {
        info.selected.push_back(col);
        info.counts[col] = values.size();
    }
    emit(info);
}

MIST_INLINE void engine_t::handle(const cmd::show_products&, emit_fn emit) {
    auto info = resp::products_info{};
    info.available = physics_.product_names();
    info.selected = state_.selected_products;
    emit(info);
}

MIST_INLINE void engine_t::handle(const cmd::show_profiler&, emit_fn emit) {
    auto data = physics_.profiler_data();
    auto info = resp::profiler_info{};

    for (const auto& [name, entry] : data) {
        info.entries.push_back({name, entry.count, entry.time});
        info.total_time += entry.time;
    }

    std::sort(info.entries.begin(), info.entries.end(),
        [](const auto& a, const auto& b) { return a.time > b.time; });

    emit(info);
}

MIST_INLINE void engine_t::handle(const cmd::show_driver&, emit_fn emit) {
    auto oss = std::ostringstream{};
    auto writer = ascii_writer{oss};
    serialize(writer, "driver_state", state_);
    emit(resp::driver_state{oss.str()});
}

MIST_INLINE void engine_t::handle(const cmd::help&, emit_fn emit) {
    emit(resp::help_text{help_text});
}

MIST_INLINE void engine_t::handle(const cmd::help_schema&, emit_fn emit) {
    emit(resp::help_text{schema_text});
}

MIST_INLINE void engine_t::handle(const cmd::stop&, emit_fn emit) {
    emit(resp::stopped{});
}

} // namespace mist::driver

#undef MIST_INLINE
