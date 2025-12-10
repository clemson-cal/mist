#pragma once

#include <string>
#include <vector>

namespace mist {

// =============================================================================
// Archive format enum
// =============================================================================

enum class archive_format {
    ascii,
    binary,
    hdf5,
};

inline auto to_string(archive_format f) -> const char* {
    switch (f) {
        case archive_format::ascii:  return "ascii";
        case archive_format::binary: return "binary";
        case archive_format::hdf5:   return "hdf5";
    }
    return "unknown";
}

inline auto from_string(std::type_identity<archive_format>, const std::string& s) -> archive_format {
    if (s == "ascii")  return archive_format::ascii;
    if (s == "binary") return archive_format::binary;
    if (s == "hdf5")   return archive_format::hdf5;
    throw std::runtime_error("unknown archive format: " + s);
}

} // namespace mist
