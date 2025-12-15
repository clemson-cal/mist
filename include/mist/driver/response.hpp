#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace mist::driver {

// =============================================================================
// Response structs
// =============================================================================

namespace resp {

// --- Status ---

struct ok {
    std::string message;
};

struct error {
    std::string what;
};

struct interrupted {};

struct stopped {};

// --- Iteration ---

struct iteration_status {
    int n;
    std::map<std::string, double> times;
    double dt;
    double zps;
};

struct timeseries_sample {
    std::map<std::string, double> values;
};

// --- Show (serialized) ---

struct physics_config {
    std::string text;
};

struct initial_config {
    std::string text;
};

struct driver_state {
    std::string text;
};

struct help_text {
    std::string text;
};

// --- Show (structured) ---

struct timeseries_info {
    std::vector<std::string> available;
    std::vector<std::string> selected;
    std::map<std::string, std::size_t> counts;
};

struct products_info {
    std::vector<std::string> available;
    std::vector<std::string> selected;
};

struct profiler_entry {
    std::string name;
    std::size_t count;
    double time;
};

struct profiler_info {
    std::vector<profiler_entry> entries;
    double total_time;
};

// --- Write ---

struct wrote_file {
    std::string filename;
    std::size_t bytes;
};

struct socket_listening {
    int port;
};

struct socket_sent {
    std::size_t bytes;
};

struct socket_cancelled {};

} // namespace resp

// =============================================================================
// response_t variant
// =============================================================================

using response_t = std::variant<
    // Status
    resp::ok,
    resp::error,
    resp::interrupted,
    resp::stopped,
    // Iteration
    resp::iteration_status,
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
    // Write
    resp::wrote_file,
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
