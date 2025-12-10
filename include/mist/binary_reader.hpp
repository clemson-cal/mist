#pragma once

#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>
#include "binary_writer.hpp"
#include "core.hpp"

namespace mist {

// =============================================================================
// Binary Reader - key-based lookup, missing fields return false
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
    explicit binary_reader(std::istream& is) : is_(is) {}

    // --- Scalars ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read_scalar(const char* name, T& value) -> bool {
        if (!seek_field(name)) return false;
        uint8_t type_tag = read_type_tag();
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
        return true;
    }

    auto read_string(const char* name, std::string& value) -> bool {
        if (!seek_field(name)) return false;
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_STRING) {
            throw std::runtime_error("Expected string type for field '" + std::string(name) + "'");
        }
        value = read_string_data();
        return true;
    }

    auto read_string(std::string& value) -> bool {
        // Anonymous read within list
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_STRING) {
            return false;
        }
        value = read_string_data();
        return true;
    }

    // --- Arrays ---

    template<typename T, std::size_t N>
    auto read_array(const char* name, vec_t<T, N>& value) -> bool {
        if (!seek_field(name)) return false;
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_ARRAY) {
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
        return true;
    }

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read_array(const char* name, std::vector<T>& value) -> bool {
        if (!seek_field(name)) return false;
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type for field '" + std::string(name) + "'");
        }
        uint8_t elem_tag = read_type_tag();
        uint64_t count;
        read_raw(count);
        value.resize(count);
        for (auto& elem : value) {
            elem = read_element<T>(elem_tag);
        }
        return true;
    }

    // --- Bulk data ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read_data(const char* name, T* ptr, std::size_t count) -> bool {
        if (!seek_field(name)) return false;
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_ARRAY) {
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
        is_.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(count * sizeof(T)));
        if (!is_) {
            throw std::runtime_error("Failed to read data for field '" + std::string(name) + "'");
        }
        return true;
    }

    // --- Groups ---

    auto begin_group(const char* name) -> bool {
        if (!seek_field(name)) return false;
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_GROUP) {
            throw std::runtime_error("Expected group type for field '" + std::string(name) + "'");
        }
        uint64_t field_count;
        read_raw(field_count);
        group_stack_.push_back({is_.tellg(), field_count, false});
        return true;
    }

    auto begin_group() -> bool {
        // Anonymous group (within list) - read type tag
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_GROUP) {
            return false;
        }
        uint64_t field_count;
        read_raw(field_count);
        // Anonymous groups within lists are still seekable (not is_list)
        group_stack_.push_back({is_.tellg(), field_count, false});
        return true;
    }

    void end_group() {
        if (!group_stack_.empty()) {
            group_stack_.pop_back();
        }
    }

    auto begin_list(const char* name) -> bool {
        if (!seek_field(name)) return false;
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_LIST) {
            throw std::runtime_error("Expected list type for field '" + std::string(name) + "'");
        }
        uint64_t item_count;
        read_raw(item_count);
        group_stack_.push_back({is_.tellg(), item_count, true});
        return true;
    }

    void end_list() { end_group(); }

    // --- Query ---

    auto has_field(const char* name) -> bool {
        auto pos = is_.tellg();
        bool found = seek_field(name);
        is_.seekg(pos);
        return found;
    }

    auto count_items(const char* name) -> std::size_t {
        auto pos = is_.tellg();
        if (!seek_field(name)) {
            is_.seekg(pos);
            return 0;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_LIST && type_tag != binary_format::TYPE_GROUP) {
            is_.seekg(pos);
            return 0;
        }
        uint64_t count;
        read_raw(count);
        is_.seekg(pos);
        return static_cast<std::size_t>(count);
    }

    // Legacy compatibility
    auto count_groups(const char* name) -> std::size_t { return count_items(name); }
    auto count_strings(const char* name) -> std::size_t { return count_items(name); }

private:
    std::istream& is_;

    struct group_info_t {
        std::streampos start;
        uint64_t remaining;
        bool is_list;
    };
    std::vector<group_info_t> group_stack_;
    bool header_read_ = false;

    void ensure_header() {
        if (header_read_) return;
        uint32_t magic;
        read_raw(magic);
        if (magic != binary_format::MAGIC) {
            throw std::runtime_error("Invalid binary archive: bad magic number");
        }
        uint8_t version;
        read_raw(version);
        if (version != binary_format::VERSION) {
            throw std::runtime_error("Unsupported binary archive version: " + std::to_string(version));
        }
        header_read_ = true;
    }

    auto read_type_tag() -> uint8_t {
        uint8_t tag;
        read_raw(tag);
        return tag;
    }

    auto read_name() -> std::string {
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

    auto read_string_data() -> std::string {
        uint64_t length;
        read_raw(length);
        std::string value(length, '\0');
        if (length > 0) {
            is_.read(value.data(), static_cast<std::streamsize>(length));
            if (!is_) {
                throw std::runtime_error("Failed to read string data");
            }
        }
        return value;
    }

    template<typename T>
    void read_raw(T& value) {
        is_.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!is_) {
            throw std::runtime_error("Failed to read data from binary archive");
        }
    }

    template<typename T>
    auto read_element(uint8_t elem_tag) -> T {
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
        }
        throw std::runtime_error("Unknown element type tag");
    }

    // Skip a field value (for seeking)
    void skip_field_value(uint8_t type_tag) {
        if (type_tag == binary_format::TYPE_FLOAT64) {
            double v;
            read_raw(v);
        } else if (type_tag == binary_format::TYPE_INT32) {
            int32_t v;
            read_raw(v);
        } else if (type_tag == binary_format::TYPE_INT64) {
            int64_t v;
            read_raw(v);
        } else if (type_tag == binary_format::TYPE_STRING) {
            uint64_t len;
            read_raw(len);
            is_.seekg(len, std::ios::cur);
        } else if (type_tag == binary_format::TYPE_ARRAY) {
            uint8_t elem_tag;
            read_raw(elem_tag);
            uint64_t count;
            read_raw(count);
            std::size_t elem_size = (elem_tag == binary_format::ELEM_INT32) ? 4 : 8;
            is_.seekg(count * elem_size, std::ios::cur);
        } else if (type_tag == binary_format::TYPE_GROUP) {
            uint64_t field_count;
            read_raw(field_count);
            for (uint64_t i = 0; i < field_count; ++i) {
                read_name();  // skip field name
                uint8_t nested_tag = read_type_tag();
                skip_field_value(nested_tag);
            }
        } else if (type_tag == binary_format::TYPE_LIST) {
            uint64_t item_count;
            read_raw(item_count);
            for (uint64_t i = 0; i < item_count; ++i) {
                uint8_t item_tag = read_type_tag();
                skip_field_value(item_tag);
            }
        }
    }

    // Seek to a field by name within the current group
    auto seek_field(const char* name) -> bool {
        ensure_header();

        auto start = group_stack_.empty() ? std::streampos(5) : group_stack_.back().start;
        is_.seekg(start);

        // If in a list, we can't seek by name - just return false
        if (!group_stack_.empty() && group_stack_.back().is_list) {
            return false;
        }

        // Scan through fields in current group
        uint64_t fields_to_scan = group_stack_.empty() ? UINT64_MAX : group_stack_.back().remaining;

        for (uint64_t i = 0; i < fields_to_scan && is_; ++i) {
            auto field_name = read_name();
            if (field_name.empty() && !is_) {
                return false;  // EOF
            }
            if (field_name == name) {
                return true;
            }
            // Skip this field's value
            uint8_t type_tag = read_type_tag();
            skip_field_value(type_tag);
        }
        return false;
    }
};

} // namespace mist
