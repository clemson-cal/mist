#pragma once

#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include "core.hpp"

namespace mist {

// =============================================================================
// ASCII Writer
// =============================================================================

class ascii_writer {
public:
    explicit ascii_writer(std::ostream& os, int indent_size = 4)
        : os_(os), indent_size_(indent_size), indent_level_(0) {}

    // =========================================================================
    // Scalar types
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_scalar(const char* name, const T& value) {
        write_indent();
        os_ << name << " = " << format_value(value) << "\n";
    }

    // =========================================================================
    // String type
    // =========================================================================

    void write_string(const char* name, const std::string& value) {
        write_indent();
        os_ << name << " = \"" << escape_string(value) << "\"\n";
    }

    // =========================================================================
    // Arrays (fixed-size vec_t)
    // =========================================================================

    template<typename T, std::size_t N>
    void write_array(const char* name, const vec_t<T, N>& value) {
        write_indent();
        os_ << name << " = [";
        for (std::size_t i = 0; i < N; ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(value[i]);
        }
        os_ << "]\n";
    }

    // =========================================================================
    // Arrays (dynamic std::vector)
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_array(const char* name, const std::vector<T>& value) {
        write_indent();
        os_ << name << " = [";
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(value[i]);
        }
        os_ << "]\n";
    }

    // =========================================================================
    // Groups (named and anonymous)
    // =========================================================================

    void begin_group(const char* name) {
        write_indent();
        os_ << name << " {\n";
        indent_level_++;
    }

    void begin_group() {
        write_indent();
        os_ << "{\n";
        indent_level_++;
    }

    void end_group() {
        indent_level_--;
        write_indent();
        os_ << "}\n";
    }

    // List operations (for vectors of compound types)
    // In ASCII format, lists are identical to groups
    void begin_list(const char* name) {
        begin_group(name);
    }

    void end_list() {
        end_group();
    }

    // =========================================================================
    // Bulk data (for ndarray) - not implemented for ASCII
    // =========================================================================

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_data(const char* /*name*/, const T* /*ptr*/, std::size_t /*count*/) {
        throw std::runtime_error("write_data not implemented for ASCII format");
    }

private:
    std::ostream& os_;
    int indent_size_;
    int indent_level_;

    void write_indent() {
        for (int i = 0; i < indent_level_ * indent_size_; ++i) {
            os_ << ' ';
        }
    }

    template<typename T>
    static std::string format_value(const T& value) {
        if constexpr (std::is_floating_point_v<T>) {
            std::ostringstream oss;
            oss << std::setprecision(15) << value;
            std::string s = oss.str();
            // Ensure floating point values have a decimal point
            if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
                s += ".0";
            }
            return s;
        } else {
            return std::to_string(value);
        }
    }

    static std::string escape_string(const std::string& s) {
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

} // namespace mist
