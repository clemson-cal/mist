#pragma once

#include <chrono>
#include <concepts>
#include <map>
#include <string>

namespace mist {
namespace perf {

// =============================================================================
// Profiler concept
// =============================================================================

struct profile_entry_t {
    std::size_t count = 0;
    double time = 0.0;
};

template<typename P>
concept Profiler = requires(P& p, const P& cp, const std::string& name) {
    p.start();
    p.record(name);
    p.clear();
    { cp.data() } -> std::same_as<std::map<std::string, profile_entry_t>>;
};

// =============================================================================
// Null profiler: no-op implementation
// =============================================================================

struct null_profiler_t {
    void start() {}
    void record(const std::string&) {}
    void clear() {}
    auto data() const -> std::map<std::string, profile_entry_t> { return {}; }
};

// =============================================================================
// High-precision profiler using steady_clock
// =============================================================================

struct profiler_t {
    using clock_t = std::chrono::steady_clock;
    using time_point_t = clock_t::time_point;

    void start() {
        start_time = clock_t::now();
    }

    void record(const std::string& name) {
        auto now = clock_t::now();
        auto elapsed = std::chrono::duration<double>(now - start_time).count();
        auto& entry = records[name];
        entry.count += 1;
        entry.time += elapsed;
        start_time = now;
    }

    void clear() {
        records.clear();
    }

    auto data() const -> std::map<std::string, profile_entry_t> {
        return records;
    }

private:
    time_point_t start_time;
    std::map<std::string, profile_entry_t> records;
};

} // namespace perf
} // namespace mist
