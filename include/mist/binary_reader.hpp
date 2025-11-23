#pragma once

#include <istream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include "core.hpp"
#include "binary_writer.hpp"  // For format constants

namespace mist {

// =============================================================================
// Binary Reader (Self-Describing Format)
// =============================================================================
//
// Binary format specification:
// - Header: uint32 magic ("MIST") + uint8 version
// - Field name: uint64 length prefix + UTF-8 bytes
// - Scalars: name + type tag (1 byte) + value (as double/int64)
// - Strings: name + type tag + uint64 length + UTF-8 bytes
// - Arrays: name + type tag + element type tag + uint64 count + elements
// - Groups: name + type tag + uint64 field count + fields
// - Lists: name + type tag + uint64 item count + items
//
// =============================================================================

class binary_reader {
public:
    explicit binary_reader(std::istream& is)
        : is_(is), header_read_(false) {}

    // =========================================================================
    // Scalar types
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void read_scalar(const char* name, T& value) {
        ensure_header();
        verify_name(name);
        
        uint8_t type_tag = read_type_tag();
        
        // In anonymous mode, we know the type from the C++ template
        if (in_anonymous_group_ > 0) {
            if constexpr (std::is_floating_point_v<T>) {
                double v;
                read_raw(v);
                value = static_cast<T>(v);
            } else if constexpr (sizeof(T) <= 4) {
                int32_t v;
                read_raw(v);
                value = static_cast<T>(v);
            } else {
                int64_t v;
                read_raw(v);
                value = static_cast<T>(v);
            }
            return;
        }
        
        if (type_tag == binary_format::TYPE_FLOAT64) {
            double v;
            read_raw(v);
            value = static_cast<T>(v);
        } else if (type_tag == binary_format::TYPE_INT32) {
            int32_t v;
            read_raw(v);
            value = static_cast<T>(v);
        } else if (type_tag == binary_format::TYPE_INT64) {
            int64_t v;
            read_raw(v);
            value = static_cast<T>(v);
        } else {
            throw std::runtime_error("Expected scalar type for field '" + std::string(name) + "'");
        }
    }

    // =========================================================================
    // String type
    // =========================================================================

    void read_string(const char* name, std::string& value) {
        ensure_header();
        verify_name(name);
        
        uint8_t type_tag = read_type_tag();
        
        if (in_anonymous_group_ == 0 && type_tag != binary_format::TYPE_STRING) {
            throw std::runtime_error("Expected string type for field '" + std::string(name) + "'");
        }
        
        uint64_t length;
        read_raw(length);
        
        value.resize(length);
        if (length > 0) {
            is_.read(value.data(), static_cast<std::streamsize>(length));
            if (!is_) {
                throw std::runtime_error("Failed to read string data");
            }
        }
    }

    // =========================================================================
    // Arrays (fixed-size vec_t)
    // =========================================================================

    template<typename T, std::size_t N>
    void read_array(const char* name, vec_t<T, N>& value) {
        ensure_header();
        verify_name(name);
        
        uint8_t type_tag = read_type_tag();
        
        if (in_anonymous_group_ == 0 && type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type for field '" + std::string(name) + "'");
        }
        
        uint8_t elem_tag = read_type_tag();
        
        uint64_t count;
        read_raw(count);
        
        if (count != N) {
            throw std::runtime_error(
                "Array size mismatch for field '" + std::string(name) + 
                "': expected " + std::to_string(N) + ", got " + std::to_string(count));
        }
        
        for (std::size_t i = 0; i < N; ++i) {
            value[i] = read_element<T>(elem_tag);
        }
    }

    // =========================================================================
    // Arrays (dynamic std::vector)
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void read_array(const char* name, std::vector<T>& value) {
        ensure_header();
        verify_name(name);
        
        uint8_t type_tag = read_type_tag();
        
        if (in_anonymous_group_ == 0 && type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type for field '" + std::string(name) + "'");
        }
        
        uint8_t elem_tag = read_type_tag();
        
        uint64_t count;
        read_raw(count);
        
        value.resize(count);
        for (auto& elem : value) {
            elem = read_element<T>(elem_tag);
        }
    }

    // =========================================================================
    // Bulk data (for ndarray)
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void read_data(const char* name, T* ptr, std::size_t count) {
        ensure_header();
        verify_name(name);

        uint8_t type_tag = read_type_tag();

        if (in_anonymous_group_ == 0 && type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type for field '" + std::string(name) + "'");
        }

        [[maybe_unused]] uint8_t elem_tag = read_type_tag();

        uint64_t n;
        read_raw(n);

        if (n != count) {
            throw std::runtime_error(
                "Data size mismatch for field '" + std::string(name) +
                "': expected " + std::to_string(count) + ", got " + std::to_string(n));
        }

        // Read raw bytes directly
        is_.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(count * sizeof(T)));
        if (!is_) {
            throw std::runtime_error("Failed to read data for field '" + std::string(name) + "'");
        }
    }

    // =========================================================================
    // Groups (named and anonymous)
    // =========================================================================

    void begin_group(const char* name) {
        ensure_header();
        verify_name(name);
        
        uint8_t type_tag = read_type_tag();
        
        if (in_anonymous_group_ == 0 && type_tag != binary_format::TYPE_GROUP) {
            throw std::runtime_error("Expected group type for field '" + std::string(name) + "'");
        }
        
        uint64_t field_count;
        read_raw(field_count);
        group_field_counts_.push_back(field_count);
    }

    void begin_group() {
        // Anonymous group within a list
        // First item has full schema, subsequent items skip schema
        if (!list_item_indices_.empty()) {
            list_item_indices_.back()++;
            bool is_first_item = (list_item_indices_.back() == 1);
            if (!is_first_item) {
                in_anonymous_group_++;
            }
        }
        group_field_counts_.push_back(0);
    }

    void end_group() {
        if (!group_field_counts_.empty()) {
            group_field_counts_.pop_back();
        }
        // Decrement anonymous counter if we're in anonymous mode
        if (in_anonymous_group_ > 0) {
            in_anonymous_group_--;
        }
    }

    // =========================================================================
    // Count groups (for deserializing vectors of compounds)
    // =========================================================================

    std::size_t count_groups(const char* name) {
        ensure_header();
        
        // Save position
        std::streampos pos = is_.tellg();
        
        // Read and verify name
        std::string field_name = read_name();
        if (field_name != name) {
            throw std::runtime_error(
                "Expected field '" + std::string(name) + "' but found '" + field_name + "'");
        }
        
        uint8_t type_tag;
        read_raw(type_tag);
        
        if (type_tag != binary_format::TYPE_LIST) {
            throw std::runtime_error("Expected list type for field '" + std::string(name) + "'");
        }
        
        uint64_t count;
        read_raw(count);
        
        // Restore position
        is_.seekg(pos);
        
        return static_cast<std::size_t>(count);
    }

    void begin_list(const char* name) {
        ensure_header();
        verify_name(name);
        
        uint8_t type_tag;
        read_raw(type_tag);
        
        if (type_tag != binary_format::TYPE_LIST) {
            throw std::runtime_error("Expected list type for field '" + std::string(name) + "'");
        }
        
        uint64_t item_count;
        read_raw(item_count);
        group_field_counts_.push_back(item_count);
        list_item_indices_.push_back(0);  // Track item index for this list
    }

    void end_list() {
        if (!list_item_indices_.empty()) {
            list_item_indices_.pop_back();
        }
        end_group();
    }

    // Check if we're at the end of the current group
    // For binary format, we track field counts, so check if we've read all fields
    bool at_group_end() {
        // In binary format, we can peek at the next bytes to check for end
        // For simplicity, we'll peek and check if next read would fail or hit end
        if (!is_) return true;
        std::streampos pos = is_.tellg();
        if (pos == std::streampos(-1)) return true;
        
        // Try to peek at the next field name length
        uint64_t length;
        is_.read(reinterpret_cast<char*>(&length), sizeof(length));
        bool at_end = !is_ || is_.eof();
        
        // Restore position
        is_.clear();
        is_.seekg(pos);
        return at_end;
    }

    // Peek at the next field name without consuming it
    std::string peek_identifier() {
        ensure_header();
        std::streampos pos = is_.tellg();
        std::string name = read_name();
        is_.seekg(pos);
        return name;
    }

private:
    std::istream& is_;
    bool header_read_;
    std::vector<uint64_t> group_field_counts_;
    std::vector<uint64_t> list_item_indices_;  // Track current item index in each list
    int in_anonymous_group_ = 0;  // Counter for nested anonymous groups

    void ensure_header() {
        if (!header_read_) {
            uint32_t magic;
            read_raw(magic);
            if (magic != binary_format::MAGIC) {
                throw std::runtime_error("Invalid binary archive: bad magic number");
            }
            
            uint8_t version;
            read_raw(version);
            if (version != binary_format::VERSION) {
                throw std::runtime_error(
                    "Unsupported binary archive version: " + std::to_string(version));
            }
            
            header_read_ = true;
        }
    }

    std::string read_name() {
        uint64_t length;
        read_raw(length);
        
        std::string name(length, '\0');
        if (length > 0) {
            is_.read(name.data(), static_cast<std::streamsize>(length));
            if (!is_) {
                throw std::runtime_error("Failed to read field name");
            }
        }
        return name;
    }

    void verify_name(const char* expected) {
        // Skip name reading inside anonymous groups (after first item)
        if (in_anonymous_group_ > 0) {
            return;
        }
        std::string actual = read_name();
        if (actual != expected) {
            throw std::runtime_error(
                "Expected field '" + std::string(expected) + "' but found '" + actual + "'");
        }
    }
    
    uint8_t read_type_tag() {
        // Skip type tag reading inside anonymous groups (after first item)
        if (in_anonymous_group_ > 0) {
            return 0;  // Return dummy value, caller should not use it
        }
        uint8_t tag;
        read_raw(tag);
        return tag;
    }

    template<typename T>
    void read_raw(T& value) {
        is_.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!is_) {
            throw std::runtime_error("Failed to read data from binary archive");
        }
    }

    template<typename T>
    T read_element(uint8_t elem_tag) {
        // In anonymous mode, use the C++ type to determine how to read
        if (in_anonymous_group_ > 0) {
            if constexpr (std::is_floating_point_v<T>) {
                double v;
                read_raw(v);
                return static_cast<T>(v);
            } else if constexpr (sizeof(T) <= 4) {
                int32_t v;
                read_raw(v);
                return static_cast<T>(v);
            } else {
                int64_t v;
                read_raw(v);
                return static_cast<T>(v);
            }
        }
        
        if (elem_tag == binary_format::ELEM_FLOAT64) {
            double v;
            read_raw(v);
            return static_cast<T>(v);
        } else if (elem_tag == binary_format::ELEM_INT32) {
            int32_t v;
            read_raw(v);
            return static_cast<T>(v);
        } else if (elem_tag == binary_format::ELEM_INT64) {
            int64_t v;
            read_raw(v);
            return static_cast<T>(v);
        } else {
            throw std::runtime_error("Unknown element type tag");
        }
    }
};

} // namespace mist
