#pragma once

#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include "state.hpp"
#include "../profiler.hpp"

namespace mist::driver {

// =============================================================================
// physics_interface_t - type-erased interface to physics module
// =============================================================================

struct physics_interface_t {
    virtual ~physics_interface_t() = default;

    // -------------------------------------------------------------------------
    // Discovery
    // -------------------------------------------------------------------------
    virtual auto time_names() const -> std::vector<std::string> = 0;
    virtual auto timeseries_names() const -> std::vector<std::string> = 0;
    virtual auto product_names() const -> std::vector<std::string> = 0;

    // -------------------------------------------------------------------------
    // State management
    // -------------------------------------------------------------------------
    virtual void init() = 0;
    virtual void reset() = 0;
    virtual auto has_state() const -> bool = 0;

    // -------------------------------------------------------------------------
    // Stepping
    // -------------------------------------------------------------------------
    virtual void advance(double dt_max = std::numeric_limits<double>::infinity()) = 0;
    virtual auto get_time(const std::string& var) const -> double = 0;
    virtual auto get_timeseries(const std::string& name) const -> double = 0;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    virtual void set_physics(const std::string& key, const std::string& value) = 0;
    virtual void set_initial(const std::string& key, const std::string& value) = 0;
    virtual void set_exec(const std::string& key, const std::string& value) = 0;

    // -------------------------------------------------------------------------
    // I/O - write operations
    // -------------------------------------------------------------------------
    virtual void write_physics(std::ostream& os, output_format fmt) = 0;
    virtual void write_initial(std::ostream& os, output_format fmt) = 0;
    virtual void write_state(std::ostream& os, output_format fmt) = 0;
    virtual void write_products(std::ostream& os, output_format fmt,
                                const std::vector<std::string>& selected) = 0;

    // -------------------------------------------------------------------------
    // I/O - read operations
    // -------------------------------------------------------------------------
    virtual auto load_physics(std::istream& is, output_format fmt) -> bool = 0;
    virtual auto load_initial(std::istream& is, output_format fmt) -> bool = 0;
    virtual auto load_state(std::istream& is, output_format fmt) -> bool = 0;

    // -------------------------------------------------------------------------
    // Profiler and performance
    // -------------------------------------------------------------------------
    virtual auto zone_count() const -> std::size_t = 0;
    virtual auto profiler_data() const -> std::map<std::string, perf::profile_entry_t> = 0;
};

} // namespace mist::driver
