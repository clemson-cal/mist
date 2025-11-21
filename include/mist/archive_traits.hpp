#pragma once

#include <fstream>
#include <string>
#include "ascii_writer.hpp"
#include "ascii_reader.hpp"
#include "binary_writer.hpp"
#include "binary_reader.hpp"

namespace mist {

// =============================================================================
// Archive format traits
// =============================================================================

// ASCII format (human-readable)
struct ascii_t {
    using writer = ascii_writer;
    using reader = ascii_reader;
    static constexpr const char* extension = ".dat";

    static writer make_writer(std::ostream& os) {
        return writer(os);
    }

    static reader make_reader(std::istream& is) {
        return reader(is);
    }
};

// Binary format (compact)
struct binary_t {
    using writer = binary_writer;
    using reader = binary_reader;
    static constexpr const char* extension = ".bin";

    static writer make_writer(std::ostream& os) {
        return writer(os);
    }

    static reader make_reader(std::istream& is) {
        return reader(is);
    }
};

// HDF5 format (hierarchical) - placeholder for future implementation
struct hdf5_t {
    // Forward declaration - implementation would go in hdf5_writer.hpp/hdf5_reader.hpp
    // using writer = hdf5_writer;
    // using reader = hdf5_reader;
    static constexpr const char* extension = ".h5";
};

} // namespace mist
