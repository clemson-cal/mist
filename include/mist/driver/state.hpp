#pragma once

#include <map>
#include <string>
#include <vector>
#include "command.hpp"
#include "../archive.hpp"

namespace mist::driver {

// =============================================================================
// state_t - checkpoint-persistent driver state
// =============================================================================

struct state_t {
    mist::format format = mist::format::ascii;
    int iteration = 0;
    int checkpoint_count = 0;
    int products_count = 0;
    int timeseries_count = 0;
    std::map<std::string, std::vector<double>> timeseries;
    std::vector<repeating_command_t> repeating_commands;
    std::vector<std::string> selected_products;
};

inline auto fields(const state_t& s) {
    return std::make_tuple(
        field("format", s.format),
        field("iteration", s.iteration),
        field("checkpoint_count", s.checkpoint_count),
        field("products_count", s.products_count),
        field("timeseries_count", s.timeseries_count),
        field("timeseries", s.timeseries),
        field("repeating_commands", s.repeating_commands),
        field("selected_products", s.selected_products)
    );
}

inline auto fields(state_t& s) {
    return std::make_tuple(
        field("format", s.format),
        field("iteration", s.iteration),
        field("checkpoint_count", s.checkpoint_count),
        field("products_count", s.products_count),
        field("timeseries_count", s.timeseries_count),
        field("timeseries", s.timeseries),
        field("repeating_commands", s.repeating_commands),
        field("selected_products", s.selected_products)
    );
}

} // namespace mist::driver
