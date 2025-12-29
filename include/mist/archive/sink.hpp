#pragma once

// Sink implementations for the parallel I/O serialization protocol.
// Provides ascii_sink and binary_sink classes.

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "format.hpp"

namespace archive {

// ============================================================================
// ascii_sink - human-readable key-value format
// ============================================================================

class ascii_sink {
public:
    explicit ascii_sink(std::ostream& stream, int indent = 4)
        : os(stream), indent_size(indent) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name = name;
    }

    // --- Scalars ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    void write(const T& value) {
        write_indent();
        write_pending_name_with_equals();
        os << format_value(value) << "\n";
    }

    void write(const std::string& value) {
        write_indent();
        write_pending_name_with_equals();
        os << "\"" << escape(value) << "\"\n";
    }

    void write(const char* value) {
        write(std::string(value));
    }

    // --- Arrays ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    void write(const std::vector<T>& value) {
        write_indent();
        write_pending_name_with_equals();
        os << "[";
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i > 0) os << ", ";
            os << format_value(value[i]);
        }
        os << "]\n";
    }

    // --- Groups ---

    void begin_group() {
        write_indent();
        if (pending_name) {
            os << pending_name << " ";
            pending_name = nullptr;
        }
        os << "{\n";
        indent_level++;
    }

    void end_group() {
        indent_level--;
        write_indent();
        os << "}\n";
    }

    void begin_list() { begin_group(); }
    void end_list() { end_group(); }

private:
    std::ostream& os;
    int indent_size;
    int indent_level = 0;
    const char* pending_name = nullptr;

    void write_indent() {
        for (int i = 0; i < indent_level * indent_size; ++i) {
            os << ' ';
        }
    }

    void write_pending_name_with_equals() {
        if (pending_name) {
            os << pending_name << " = ";
            pending_name = nullptr;
        }
    }

    template <typename T>
    static auto format_value(const T& value) -> std::string {
        if constexpr (std::is_floating_point_v<T>) {
            std::ostringstream oss;
            oss << std::setprecision(15) << value;
            auto s = oss.str();
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
                s += ".0";
            }
            return s;
        } else {
            return std::to_string(value);
        }
    }

    static auto escape(const std::string& s) -> std::string {
        std::string result;
        result.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '\\': result += "\\\\"; break;
                case '"':  result += "\\\""; break;
                case '\n': result += "\\n"; break;
                case '\t': result += "\\t"; break;
                case '\r': result += "\\r"; break;
                default:   result += c; break;
            }
        }
        return result;
    }
};

// ============================================================================
// binary_sink - self-describing binary format
// ============================================================================

class binary_sink {
public:
    explicit binary_sink(std::ostream& stream, bool skip_header = false)
        : os(stream), header_written(skip_header) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name = name;
    }

    // --- Scalars ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    void write(const T& value) {
        ensure_header();
        write_pending_name();

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

    void write(const std::string& value) {
        ensure_header();
        write_pending_name();
        write_type_tag(binary_format::TYPE_STRING);

        uint64_t length = value.size();
        write_raw(length);
        if (length > 0) {
            os.write(value.data(), static_cast<std::streamsize>(length));
        }
    }

    void write(const char* value) {
        write(std::string(value));
    }

    // --- Arrays ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    void write(const std::vector<T>& value) {
        ensure_header();
        write_pending_name();
        write_type_tag(binary_format::TYPE_ARRAY);
        write_type_tag(binary_format::element_type_tag<T>());

        uint64_t count = value.size();
        write_raw(count);

        for (const auto& elem : value) {
            write_element(elem);
        }
    }

    // --- Groups ---

    void begin_group() {
        ensure_header();

        if (pending_name) {
            write_name(pending_name);
            pending_name = nullptr;
            write_type_tag(binary_format::TYPE_GROUP);
            if (!field_counts.empty()) {
                field_counts.back()++;
            }
        } else {
            if (!field_counts.empty()) {
                field_counts.back()++;
            }
            write_raw(binary_format::TYPE_GROUP);
        }

        group_positions.push_back(os.tellp());
        uint64_t placeholder = 0;
        write_raw(placeholder);
        field_counts.push_back(0);
    }

    void end_group() {
        if (group_positions.empty()) {
            return;
        }

        std::streampos pos = group_positions.back();
        group_positions.pop_back();

        uint64_t count = field_counts.back();
        field_counts.pop_back();

        std::streampos current_pos = os.tellp();
        os.seekp(pos);
        write_raw(count);
        os.seekp(current_pos);
    }

    void begin_list() {
        ensure_header();

        if (pending_name) {
            write_name(pending_name);
            pending_name = nullptr;
            write_type_tag(binary_format::TYPE_LIST);
            if (!field_counts.empty()) {
                field_counts.back()++;
            }
        } else {
            if (!field_counts.empty()) {
                field_counts.back()++;
            }
            write_raw(binary_format::TYPE_LIST);
        }

        group_positions.push_back(os.tellp());
        uint64_t placeholder = 0;
        write_raw(placeholder);
        field_counts.push_back(0);
    }

    void end_list() { end_group(); }

private:
    std::ostream& os;
    bool header_written;
    std::vector<std::streampos> group_positions;
    std::vector<uint64_t> field_counts;
    const char* pending_name = nullptr;

    void ensure_header() {
        if (!header_written) {
            write_raw(binary_format::MAGIC);
            write_raw(binary_format::VERSION);
            header_written = true;
        }
    }

    void write_pending_name() {
        if (pending_name) {
            write_name(pending_name);
            pending_name = nullptr;
        }
        if (!field_counts.empty()) {
            field_counts.back()++;
        }
    }

    void write_name(const char* name) {
        uint64_t length = std::strlen(name);
        write_raw(length);
        if (length > 0) {
            os.write(name, static_cast<std::streamsize>(length));
        }
    }

    void write_type_tag(uint8_t tag) {
        write_raw(tag);
    }

    template <typename T>
    void write_raw(const T& value) {
        os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
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

} // namespace archive
