#pragma once

#include <map>
#include <string>
#include <vector>
#include "command.hpp"
#include "../serialize.hpp"

namespace mist::driver {

// =============================================================================
// Output format enum
// =============================================================================

enum class output_format {
    ascii,
    binary,
    hdf5
};

inline auto to_string(output_format fmt) -> const char* {
    switch (fmt) {
        case output_format::ascii: return "ascii";
        case output_format::binary: return "binary";
        case output_format::hdf5: return "hdf5";
    }
    return "unknown";
}

inline auto from_string(std::type_identity<output_format>, const std::string& s) -> output_format {
    if (s == "ascii") return output_format::ascii;
    if (s == "binary") return output_format::binary;
    if (s == "hdf5") return output_format::hdf5;
    throw std::runtime_error("unknown output format: " + s);
}

inline auto infer_format_from_filename(std::string_view filename) -> output_format {
    if (filename.ends_with(".dat") || filename.ends_with(".cfg")) return output_format::ascii;
    if (filename.ends_with(".bin")) return output_format::binary;
    if (filename.ends_with(".h5")) return output_format::hdf5;
    throw std::runtime_error(std::string("cannot infer format from filename: ") + std::string(filename));
}

// =============================================================================
// state_t - checkpoint-persistent driver state
// =============================================================================

struct state_t {
    output_format format = output_format::ascii;
    int iteration = 0;
    int checkpoint_count = 0;
    int products_count = 0;
    int timeseries_count = 0;
    std::map<std::string, std::vector<double>> timeseries;
    std::vector<repeating_command_t> repeating_commands;
    std::vector<std::string> selected_products;

    auto fields() const {
        return std::make_tuple(
            field("format", format),
            field("iteration", iteration),
            field("checkpoint_count", checkpoint_count),
            field("products_count", products_count),
            field("timeseries_count", timeseries_count),
            field("timeseries", timeseries),
            field("repeating_commands", repeating_commands),
            field("selected_products", selected_products)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("format", format),
            field("iteration", iteration),
            field("checkpoint_count", checkpoint_count),
            field("products_count", products_count),
            field("timeseries_count", timeseries_count),
            field("timeseries", timeseries),
            field("repeating_commands", repeating_commands),
            field("selected_products", selected_products)
        );
    }
};

} // namespace mist::driver
