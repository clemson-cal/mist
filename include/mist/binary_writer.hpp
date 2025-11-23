#pragma once

#include <ostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include "core.hpp"

namespace mist {

// =============================================================================
// Binary Format Type Tags
// =============================================================================

namespace binary_format {
    // Magic header to identify mist binary archives
    constexpr uint32_t MAGIC = 0x4D495354;  // "MIST" in ASCII
    constexpr uint8_t VERSION = 1;

    // Type tags
    constexpr uint8_t TYPE_INT32   = 0x01;
    constexpr uint8_t TYPE_INT64   = 0x02;
    constexpr uint8_t TYPE_FLOAT64 = 0x03;
    constexpr uint8_t TYPE_STRING  = 0x04;
    constexpr uint8_t TYPE_ARRAY   = 0x05;
    constexpr uint8_t TYPE_GROUP   = 0x06;
    constexpr uint8_t TYPE_LIST    = 0x07;  // Anonymous groups (vector of compounds)

    // Element type tags for arrays
    constexpr uint8_t ELEM_INT32   = 0x01;
    constexpr uint8_t ELEM_INT64   = 0x02;
    constexpr uint8_t ELEM_FLOAT64 = 0x03;

    template<typename T>
    constexpr uint8_t scalar_type_tag() {
        if constexpr (std::is_same_v<T, int32_t> || (std::is_same_v<T, int> && sizeof(int) == 4)) {
            return TYPE_INT32;
        } else if constexpr (std::is_same_v<T, int64_t> || (std::is_same_v<T, long> && sizeof(long) == 8)) {
            return TYPE_INT64;
        } else if constexpr (std::is_floating_point_v<T>) {
            return TYPE_FLOAT64;
        } else {
            static_assert(std::is_arithmetic_v<T>, "Unsupported scalar type");
            // Default to int64 for other integer types
            return TYPE_INT64;
        }
    }

    template<typename T>
    constexpr uint8_t element_type_tag() {
        if constexpr (std::is_floating_point_v<T>) {
            return ELEM_FLOAT64;
        } else if constexpr (sizeof(T) <= 4) {
            return ELEM_INT32;
        } else {
            return ELEM_INT64;
        }
    }
}

// =============================================================================
// Binary Writer (Self-Describing Format)
// =============================================================================
//
// Binary format specification:
// - Header: uint32 magic ("MIST") + uint8 version
// - Field name: uint64 length prefix + UTF-8 bytes
// - Scalars: name + type tag (1 byte) + value (as double/int64)
// - Strings: name + type tag + uint64 length + UTF-8 bytes
// - Arrays: name + type tag + element type tag + uint64 count + elements
// - Groups: name + type tag + uint64 field count + fields
// - Lists: name + type tag + uint64 item count + items (each is anonymous group)
//
// =============================================================================

class binary_writer {
public:
    explicit binary_writer(std::ostream& os)
        : os_(os), header_written_(false) {}

    // =========================================================================
    // Scalar types
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_scalar(const char* name, const T& value) {
        ensure_header();
        write_name(name);
        
        // Write type tag and value (promote to standard sizes)
        if constexpr (std::is_floating_point_v<T>) {
            write_type_tag(binary_format::TYPE_FLOAT64);
            double v = static_cast<double>(value);
            write_raw(v);
        } else if constexpr (sizeof(T) <= 4) {
            write_type_tag(binary_format::TYPE_INT32);
            int32_t v = static_cast<int32_t>(value);
            write_raw(v);
        } else {
            write_type_tag(binary_format::TYPE_INT64);
            int64_t v = static_cast<int64_t>(value);
            write_raw(v);
        }
    }

    // =========================================================================
    // String type
    // =========================================================================

    void write_string(const char* name, const std::string& value) {
        ensure_header();
        write_name(name);
        write_type_tag(binary_format::TYPE_STRING);
        
        uint64_t length = value.size();
        write_raw(length);
        if (length > 0) {
            os_.write(value.data(), static_cast<std::streamsize>(length));
        }
    }

    // =========================================================================
    // Arrays (fixed-size vec_t)
    // =========================================================================

    template<typename T, std::size_t N>
    void write_array(const char* name, const vec_t<T, N>& value) {
        ensure_header();
        write_name(name);
        write_type_tag(binary_format::TYPE_ARRAY);
        write_type_tag(binary_format::element_type_tag<T>());
        
        uint64_t count = N;
        write_raw(count);
        
        // Write elements in standardized format
        for (std::size_t i = 0; i < N; ++i) {
            write_element(value[i]);
        }
    }

    // =========================================================================
    // Arrays (dynamic std::vector)
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_array(const char* name, const std::vector<T>& value) {
        ensure_header();
        write_name(name);
        write_type_tag(binary_format::TYPE_ARRAY);
        write_type_tag(binary_format::element_type_tag<T>());
        
        uint64_t count = value.size();
        write_raw(count);
        
        for (const auto& elem : value) {
            write_element(elem);
        }
    }

    // =========================================================================
    // Bulk data (for ndarray)
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_data(const char* name, const T* ptr, std::size_t count) {
        ensure_header();
        write_name(name);
        write_type_tag(binary_format::TYPE_ARRAY);
        write_type_tag(binary_format::element_type_tag<T>());

        uint64_t n = count;
        write_raw(n);

        // Write raw bytes directly (no per-element conversion)
        os_.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(count * sizeof(T)));
    }

    // =========================================================================
    // Groups (named and anonymous)
    // =========================================================================

    void begin_group(const char* name) {
        ensure_header();
        write_name(name);
        write_type_tag(binary_format::TYPE_GROUP);
        
        // Save position for field count backfill
        group_positions_.push_back(os_.tellp());
        uint64_t placeholder = 0;
        write_raw(placeholder);
        field_counts_.push_back(0);
    }

    void begin_group() {
        // Anonymous group within a list - increment parent's count
        if (!field_counts_.empty()) {
            field_counts_.back()++;
        }
        // Push sentinel for anonymous group (no position tracking needed)
        group_positions_.push_back(-1);
        field_counts_.push_back(0);
        
        // For anonymous groups: first item writes full schema, subsequent items skip it
        // Check if this is the first item (parent count was just incremented to 1)
        bool is_first_item = false;
        if (field_counts_.size() >= 2) {
            // Parent's count is at field_counts_[size-2], we just incremented it
            is_first_item = (field_counts_[field_counts_.size() - 2] == 1);
        }
        
        if (!is_first_item) {
            in_anonymous_group_++;
        }
    }

    void begin_list(const char* name) {
        ensure_header();
        write_name(name);
        write_type_tag(binary_format::TYPE_LIST);
        
        // Save position for item count backfill
        group_positions_.push_back(os_.tellp());
        uint64_t placeholder = 0;
        write_raw(placeholder);
        field_counts_.push_back(0);
    }

    void end_list() {
        end_group();
    }

    void end_group() {
        if (group_positions_.empty()) {
            return;
        }

        std::streampos pos = group_positions_.back();
        group_positions_.pop_back();
        
        uint64_t count = field_counts_.back();
        field_counts_.pop_back();

        // If this was an anonymous group (sentinel -1)
        if (pos == std::streampos(-1)) {
            // Decrement anonymous counter if we're in anonymous mode
            if (in_anonymous_group_ > 0) {
                in_anonymous_group_--;
            }
        } else {
            // Named group/list - backfill the count
            std::streampos current_pos = os_.tellp();
            os_.seekp(pos);
            write_raw(count);
            os_.seekp(current_pos);
        }
    }

private:
    std::ostream& os_;
    bool header_written_;
    std::vector<std::streampos> group_positions_;
    std::vector<uint64_t> field_counts_;
    int in_anonymous_group_ = 0;  // Counter for nested anonymous groups

    void ensure_header() {
        if (!header_written_) {
            write_raw(binary_format::MAGIC);
            write_raw(binary_format::VERSION);
            header_written_ = true;
        }
        // Increment field count for parent group (only if not in anonymous group)
        if (!field_counts_.empty() && in_anonymous_group_ == 0) {
            field_counts_.back()++;
        }
    }

    void write_name(const char* name) {
        // Skip field names inside anonymous groups (list items)
        if (in_anonymous_group_ > 0) {
            return;
        }
        uint64_t length = std::strlen(name);
        write_raw(length);
        if (length > 0) {
            os_.write(name, static_cast<std::streamsize>(length));
        }
    }
    
    void write_type_tag(uint8_t tag) {
        // Skip type tags inside anonymous groups (structure is known from first item)
        if (in_anonymous_group_ > 0) {
            return;
        }
        write_raw(tag);
    }

    template<typename T>
    void write_raw(const T& value) {
        os_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template<typename T>
    void write_element(const T& value) {
        if constexpr (std::is_floating_point_v<T>) {
            double v = static_cast<double>(value);
            write_raw(v);
        } else if constexpr (sizeof(T) <= 4) {
            int32_t v = static_cast<int32_t>(value);
            write_raw(v);
        } else {
            int64_t v = static_cast<int64_t>(value);
            write_raw(v);
        }
    }
};

} // namespace mist
