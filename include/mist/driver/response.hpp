#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <variant>
#include <vector>
#include "../archive.hpp"

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

inline auto enabled() -> scheme_t { return scheme_t{}; }
inline auto disabled() -> scheme_t {
    return scheme_t{"", "", "", "", "", "", "", "", "", "", "", ""};
}

inline auto is_tty(int fd) -> bool { return isatty(fd) != 0; }
inline auto is_tty(std::ostream& os) -> bool {
    if (&os == &std::cout) return is_tty(STDOUT_FILENO);
    if (&os == &std::cerr) return is_tty(STDERR_FILENO);
    return false;
}

inline auto for_stream(std::ostream& os) -> scheme_t {
    return is_tty(os) ? enabled() : disabled();
}

} // namespace color

// =============================================================================
// Response structs
// =============================================================================

namespace resp {

// --- Status ---

struct ok {
    std::string message;
};
inline auto fields(const ok& s) { return std::make_tuple(field("message", s.message)); }
inline auto fields(ok& s) { return std::make_tuple(field("message", s.message)); }

struct error {
    std::string what;
};
inline auto fields(const error& s) { return std::make_tuple(field("what", s.what)); }
inline auto fields(error& s) { return std::make_tuple(field("what", s.what)); }

struct interrupted {};
inline auto fields(const interrupted&) { return std::make_tuple(); }
inline auto fields(interrupted&) { return std::make_tuple(); }

struct stopped {};
inline auto fields(const stopped&) { return std::make_tuple(); }
inline auto fields(stopped&) { return std::make_tuple(); }

struct state_info {
    bool initialized;
    std::size_t zone_count;
    std::map<std::string, double> times;
};
inline auto fields(const state_info& s) {
    return std::make_tuple(
        field("initialized", s.initialized),
        field("zone_count", s.zone_count),
        field("times", s.times)
    );
}
inline auto fields(state_info& s) {
    return std::make_tuple(
        field("initialized", s.initialized),
        field("zone_count", s.zone_count),
        field("times", s.times)
    );
}

// --- Iteration ---

struct iteration_info {
    int n;
    std::map<std::string, double> times;
    double dt;
    double zps;
};
inline auto fields(const iteration_info& s) {
    return std::make_tuple(
        field("n", s.n),
        field("times", s.times),
        field("dt", s.dt),
        field("zps", s.zps)
    );
}
inline auto fields(iteration_info& s) {
    return std::make_tuple(
        field("n", s.n),
        field("times", s.times),
        field("dt", s.dt),
        field("zps", s.zps)
    );
}

using timeseries_sample = std::map<std::string, double>;

// --- Show (serialized) ---

struct physics_config {
    std::string text;
};
inline auto fields(const physics_config& s) { return std::make_tuple(field("text", s.text)); }
inline auto fields(physics_config& s) { return std::make_tuple(field("text", s.text)); }

struct initial_config {
    std::string text;
};
inline auto fields(const initial_config& s) { return std::make_tuple(field("text", s.text)); }
inline auto fields(initial_config& s) { return std::make_tuple(field("text", s.text)); }

struct driver_state {
    std::string text;
};
inline auto fields(const driver_state& s) { return std::make_tuple(field("text", s.text)); }
inline auto fields(driver_state& s) { return std::make_tuple(field("text", s.text)); }

struct help_text {
    std::string text;
};
inline auto fields(const help_text& s) { return std::make_tuple(field("text", s.text)); }
inline auto fields(help_text& s) { return std::make_tuple(field("text", s.text)); }

// --- Show (structured) ---

struct timeseries_info {
    std::vector<std::string> available;
    std::vector<std::string> selected;
    std::map<std::string, std::size_t> counts;
};
inline auto fields(const timeseries_info& s) {
    return std::make_tuple(
        field("available", s.available),
        field("selected", s.selected),
        field("counts", s.counts)
    );
}
inline auto fields(timeseries_info& s) {
    return std::make_tuple(
        field("available", s.available),
        field("selected", s.selected),
        field("counts", s.counts)
    );
}

struct products_info {
    std::vector<std::string> available;
    std::vector<std::string> selected;
};
inline auto fields(const products_info& s) {
    return std::make_tuple(
        field("available", s.available),
        field("selected", s.selected)
    );
}
inline auto fields(products_info& s) {
    return std::make_tuple(
        field("available", s.available),
        field("selected", s.selected)
    );
}

struct profiler_entry {
    std::string name;
    std::size_t count;
    double time;
};
inline auto fields(const profiler_entry& s) {
    return std::make_tuple(
        field("name", s.name),
        field("count", s.count),
        field("time", s.time)
    );
}
inline auto fields(profiler_entry& s) {
    return std::make_tuple(
        field("name", s.name),
        field("count", s.count),
        field("time", s.time)
    );
}

struct profiler_info {
    std::vector<profiler_entry> entries;
    double total_time;
};
inline auto fields(const profiler_info& s) {
    return std::make_tuple(
        field("entries", s.entries),
        field("total_time", s.total_time)
    );
}
inline auto fields(profiler_info& s) {
    return std::make_tuple(
        field("entries", s.entries),
        field("total_time", s.total_time)
    );
}

struct exec_info {
    int threads;
    int mpi_rank;
    int mpi_size;
};
inline auto fields(const exec_info& s) {
    return std::make_tuple(
        field("threads", s.threads),
        field("mpi_rank", s.mpi_rank),
        field("mpi_size", s.mpi_size)
    );
}
inline auto fields(exec_info& s) {
    return std::make_tuple(
        field("threads", s.threads),
        field("mpi_rank", s.mpi_rank),
        field("mpi_size", s.mpi_size)
    );
}

struct build_info {
    std::string version;
    std::string build_type;
    std::string compiler;
    std::string platform;
    std::string git_commit;
    std::string git_branch;
    bool git_dirty;
    bool with_mpi;
};
inline auto fields(const build_info& s) {
    return std::make_tuple(
        field("version", s.version),
        field("build_type", s.build_type),
        field("compiler", s.compiler),
        field("platform", s.platform),
        field("git_commit", s.git_commit),
        field("git_branch", s.git_branch),
        field("git_dirty", s.git_dirty),
        field("with_mpi", s.with_mpi)
    );
}
inline auto fields(build_info& s) {
    return std::make_tuple(
        field("version", s.version),
        field("build_type", s.build_type),
        field("compiler", s.compiler),
        field("platform", s.platform),
        field("git_commit", s.git_commit),
        field("git_branch", s.git_branch),
        field("git_dirty", s.git_dirty),
        field("with_mpi", s.with_mpi)
    );
}

// --- Write ---

struct wrote_file {
    std::string filename;
    std::size_t bytes;
};
inline auto fields(const wrote_file& s) {
    return std::make_tuple(
        field("filename", s.filename),
        field("bytes", s.bytes)
    );
}
inline auto fields(wrote_file& s) {
    return std::make_tuple(
        field("filename", s.filename),
        field("bytes", s.bytes)
    );
}

// --- Socket ---

struct socket_listening {
    int port;
};
inline auto fields(const socket_listening& s) { return std::make_tuple(field("port", s.port)); }
inline auto fields(socket_listening& s) { return std::make_tuple(field("port", s.port)); }

struct socket_sent {
    std::size_t bytes;
};
inline auto fields(const socket_sent& s) { return std::make_tuple(field("bytes", s.bytes)); }
inline auto fields(socket_sent& s) { return std::make_tuple(field("bytes", s.bytes)); }

struct socket_cancelled {};
inline auto fields(const socket_cancelled&) { return std::make_tuple(); }
inline auto fields(socket_cancelled&) { return std::make_tuple(); }

} // namespace resp

// =============================================================================
// Print functions for human-readable output
// =============================================================================

inline void print(std::ostream& os, const color::scheme_t& c, const resp::ok& r) {
    os << c.info << r.message << c.reset << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::error& r) {
    os << c.error << "error: " << c.reset << r.what << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::interrupted&) {
    os << "\n" << c.warning << "interrupted" << c.reset << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::stopped&) {
    os << "\n" << c.header << "=== Session Complete ===" << c.reset << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::state_info& r) {
    os << c.label << "physics state: " << c.reset;
    if (r.initialized) {
        os << c.selected << "initialized" << c.reset;
        os << " (" << c.value << r.zone_count << c.reset << " zones)";
        for (const auto& [name, value] : r.times) {
            os << " " << c.label << name << "=" << c.reset
               << c.value << std::scientific << std::setprecision(6) << value << c.reset;
        }
    } else {
        os << c.unselected << "none" << c.reset;
    }
    os << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::iteration_info& r) {
    os << c.iteration << "[" << std::setw(6) << std::setfill('0')
       << r.n << std::setfill(' ') << "]" << c.reset << " ";

    for (const auto& [name, value] : r.times) {
        os << c.label << name << "=" << c.reset
           << c.value << std::scientific << std::showpos << std::setprecision(6)
           << value << std::noshowpos << c.reset << " ";
    }

    os << c.label << "dt=" << c.reset
       << c.value << std::scientific << std::setprecision(6) << r.dt << c.reset << " ";
    os << c.label << "zps=" << c.reset
       << c.value << std::scientific << std::setprecision(2) << r.zps << c.reset << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::timeseries_sample& r) {
    os << c.info << "recorded sample" << c.reset << " (";
    auto first = true;
    for (const auto& [name, value] : r) {
        if (!first) os << ", ";
        os << c.label << name << "=" << c.reset
           << c.value << std::scientific << std::setprecision(6) << value << c.reset;
        first = false;
    }
    os << ")\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::physics_config& r) {
    os << r.text;
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::initial_config& r) {
    os << r.text;
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::driver_state& r) {
    os << r.text;
}

inline void print(std::ostream& os, const color::scheme_t&, const resp::help_text& r) {
    os << r.text << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::timeseries_info& r) {
    os << c.header << "Timeseries:" << c.reset << "\n";
    for (const auto& col : r.available) {
        auto it = std::find(r.selected.begin(), r.selected.end(), col);
        auto is_selected = (it != r.selected.end());
        if (is_selected) {
            auto count_it = r.counts.find(col);
            auto count = (count_it != r.counts.end()) ? count_it->second : 0;
            os << "  - " << c.selected << "[+]" << c.reset << " "
               << c.key << col << c.reset
               << " (" << c.value << count << c.reset << " samples)\n";
        } else {
            os << "  - " << c.unselected << "[ ] " << col << c.reset << "\n";
        }
    }
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::products_info& r) {
    os << c.header << "Products:" << c.reset << "\n";
    for (const auto& prod : r.available) {
        auto is_selected = std::find(r.selected.begin(), r.selected.end(), prod) != r.selected.end();
        if (is_selected) {
            os << "  - " << c.selected << "[+]" << c.reset << " "
               << c.key << prod << c.reset << "\n";
        } else {
            os << "  - " << c.unselected << "[ ] " << prod << c.reset << "\n";
        }
    }
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::exec_info& r) {
    os << c.header << "Execution context:" << c.reset << "\n";
    os << "  " << c.label << "threads: " << c.reset << c.value << r.threads << c.reset << "\n";
    os << "  " << c.label << "mpi_size: " << c.reset << c.value << r.mpi_size << c.reset << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::build_info& r) {
    os << c.header << "Build info:" << c.reset << "\n";
    os << "  " << c.label << "version: " << c.reset << c.value << r.version << c.reset << "\n";
    if (!r.build_type.empty()) {
        os << "  " << c.label << "build_type: " << c.reset << c.value << r.build_type << c.reset << "\n";
    }
    os << "  " << c.label << "compiler: " << c.reset << c.value << r.compiler << c.reset << "\n";
    os << "  " << c.label << "platform: " << c.reset << c.value << r.platform << c.reset << "\n";
    os << "  " << c.label << "git: " << c.reset << c.value << r.git_commit << c.reset;
    if (!r.git_branch.empty() && r.git_branch != "unknown") {
        os << " (" << c.value << r.git_branch << c.reset << ")";
    }
    if (r.git_dirty) {
        os << " " << c.warning << "[dirty]" << c.reset;
    }
    os << "\n";
    os << "  " << c.label << "mpi: " << c.reset << c.value << (r.with_mpi ? "enabled" : "disabled") << c.reset << "\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::profiler_info& r) {
    if (r.entries.empty()) return;

    const int col_stage = 24;
    const int col_count = 10;
    const int col_time = 12;
    const int col_pct = 8;
    const int table_width = col_stage + col_count + col_time + col_pct + 7;
    auto sep = std::string(table_width, '-');

    os << "\n" << c.header << "Profiler" << c.reset << "\n";
    os << c.unselected << sep << c.reset << "\n";

    os << c.label
       << std::left << std::setw(col_stage) << " stage"
       << std::right << std::setw(col_count) << "count"
       << std::setw(col_time) << "time[s]"
       << std::setw(col_pct) << "%"
       << c.reset << "\n";

    os << c.unselected << sep << c.reset << "\n";

    auto sorted = r.entries;
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.time > b.time; });

    for (const auto& entry : sorted) {
        auto pct = (r.total_time > 0) ? 100.0 * entry.time / r.total_time : 0.0;
        os << " " << c.value << std::left << std::setw(col_stage - 1) << entry.name << c.reset
           << c.iteration << std::right << std::setw(col_count) << entry.count << c.reset
           << c.info << std::setw(col_time) << std::fixed << std::setprecision(4) << entry.time << c.reset
           << c.warning << std::setw(col_pct - 1) << std::fixed << std::setprecision(1) << pct << "%" << c.reset
           << "\n";
    }

    os << c.unselected << sep << c.reset << "\n";

    auto total_count = std::size_t{0};
    for (const auto& e : sorted) total_count += e.count;

    os << c.header
       << std::left << std::setw(col_stage) << " TOTAL"
       << std::right << std::setw(col_count) << total_count
       << std::setw(col_time) << std::fixed << std::setprecision(4) << r.total_time
       << std::setw(col_pct) << "100.0%"
       << c.reset << "\n";

    os << c.unselected << sep << c.reset << "\n\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::wrote_file& r) {
    os << c.info << "wrote " << c.value << r.filename
       << c.reset << " (" << r.bytes << " bytes)\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::socket_listening& r) {
    os << c.info << "listening on port " << c.value << r.port << c.reset << "...\n";
}

inline void print(std::ostream& os, const color::scheme_t& c, const resp::socket_sent& r) {
    os << c.info << "sent " << c.value << r.bytes << c.reset << " bytes\n";
}

inline void print(std::ostream& os, const color::scheme_t&, const resp::socket_cancelled&) {
    os << "cancelled\n";
}

// =============================================================================
// response_t variant
// =============================================================================

using response_t = std::variant<
    // Status
    resp::ok,
    resp::error,
    resp::interrupted,
    resp::stopped,
    resp::state_info,
    // Iteration
    resp::iteration_info,
    resp::timeseries_sample,
    // Show (serialized)
    resp::physics_config,
    resp::initial_config,
    resp::driver_state,
    resp::help_text,
    // Show (structured)
    resp::timeseries_info,
    resp::products_info,
    resp::profiler_info,
    resp::exec_info,
    resp::build_info,
    // Write
    resp::wrote_file,
    // Socket
    resp::socket_listening,
    resp::socket_sent,
    resp::socket_cancelled
>;

// =============================================================================
// Helper to check if response indicates an error
// =============================================================================

inline auto is_error(const response_t& r) -> bool {
    return std::holds_alternative<resp::error>(r);
}

inline auto is_stop(const response_t& r) -> bool {
    return std::holds_alternative<resp::stopped>(r);
}

} // namespace mist::driver
