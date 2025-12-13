#pragma once

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include "core.hpp"
#include "ndarray.hpp"

namespace mist {

// =============================================================================
// ASCII Writer - writes key-based format
// =============================================================================

class ascii_writer {
public:
    explicit ascii_writer(std::ostream& os, int indent_size = 4)
        : os_(os), indent_size_(indent_size), indent_level_(0) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name_ = name;
    }

    // --- Scalars ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write(const T& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << format_value(value) << "\n";
    }

    void write(const std::string& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "\"" << escape(value) << "\"\n";
    }

    void write(const char* value) {
        write(std::string(value));
    }

    // --- Arrays ---

    template<typename T, std::size_t N>
    void write(const vec_t<T, N>& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "[";
        for (std::size_t i = 0; i < N; ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(value[i]);
        }
        os_ << "]\n";
    }

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write(const std::vector<T>& value) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "[";
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(value[i]);
        }
        os_ << "]\n";
    }

    // --- Bulk data ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    void write_data(const T* ptr, std::size_t count) {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " = ";
            pending_name_ = nullptr;
        }
        os_ << "[";
        for (std::size_t i = 0; i < count; ++i) {
            if (i > 0) os_ << ", ";
            os_ << format_value(ptr[i]);
        }
        os_ << "]\n";
    }

    // Overload for vec_t elements: flatten to scalar array
    template<typename T, std::size_t N>
        requires std::is_arithmetic_v<T>
    void write_data(const vec_t<T, N>* ptr, std::size_t count) {
        write_data(reinterpret_cast<const T*>(ptr), count * N);
    }

    // --- CachedNdArray ---

    template<CachedNdArray T>
    void write(const T& arr) {
        begin_group();
        begin_named("start");
        write(start(arr));
        begin_named("shape");
        write(shape(arr));
        begin_named("data");
        write_data(data(arr), size(arr));
        end_group();
    }

    // --- Groups ---

    void begin_group() {
        write_indent();
        if (pending_name_) {
            os_ << pending_name_ << " ";
            pending_name_ = nullptr;
        }
        os_ << "{\n";
        indent_level_++;
    }

    void end_group() {
        indent_level_--;
        write_indent();
        os_ << "}\n";
    }

    void begin_list() { begin_group(); }
    void end_list() { end_group(); }

private:
    std::ostream& os_;
    int indent_size_;
    int indent_level_;
    const char* pending_name_ = nullptr;

    void write_indent() {
        for (int i = 0; i < indent_level_ * indent_size_; ++i) {
            os_ << ' ';
        }
    }

    template<typename T>
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

} // namespace mist
