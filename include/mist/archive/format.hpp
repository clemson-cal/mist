#pragma once

// Binary format constants for the parallel I/O serialization protocol.

#include <cstdint>
#include <type_traits>

namespace archive {

// ============================================================================
// Binary format constants (wire format specification)
// ============================================================================

namespace binary_format {

constexpr uint32_t MAGIC = 0x4D495354;  // "MIST" in ASCII
constexpr uint8_t VERSION = 1;

// Type tags
constexpr uint8_t TYPE_INT32   = 0x01;
constexpr uint8_t TYPE_INT64   = 0x02;
constexpr uint8_t TYPE_FLOAT64 = 0x03;
constexpr uint8_t TYPE_STRING  = 0x04;
constexpr uint8_t TYPE_ARRAY   = 0x05;
constexpr uint8_t TYPE_GROUP   = 0x06;
constexpr uint8_t TYPE_LIST    = 0x07;

// Element type tags for arrays
constexpr uint8_t ELEM_INT32   = 0x01;
constexpr uint8_t ELEM_INT64   = 0x02;
constexpr uint8_t ELEM_FLOAT64 = 0x03;

template <typename T>
constexpr uint8_t element_type_tag() {
    if constexpr (std::is_floating_point_v<T>) {
        return ELEM_FLOAT64;
    } else if constexpr (sizeof(T) <= 4) {
        return ELEM_INT32;
    } else {
        return ELEM_INT64;
    }
}

} // namespace binary_format

} // namespace archive
