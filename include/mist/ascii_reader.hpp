#pragma once

#include <cctype>
#include <istream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "core.hpp"
#include "ndarray.hpp"

namespace mist {

// =============================================================================
// ASCII Reader - key-based lookup, missing fields return false
// =============================================================================

class ascii_reader {
public:
    explicit ascii_reader(std::istream& is) : is_(is) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name_ = name;
    }

    // --- Scalars (return false if field missing) ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read(T& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        value = read_number<T>();
        return true;
    }

    auto read(std::string& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        } else {
            skip_ws();
            if (peek() != '"') return false;
        }
        value = read_quoted_string();
        return true;
    }

    // --- Arrays ---

    template<typename T, std::size_t N>
    auto read(vec_t<T, N>& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        expect('[');
        for (std::size_t i = 0; i < N; ++i) {
            skip_ws();
            value[i] = read_number<T>();
            skip_ws();
            if (i < N - 1) expect(',');
        }
        expect(']');
        return true;
    }

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read(std::vector<T>& value) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        expect('[');
        value.clear();
        skip_ws();
        if (peek() != ']') {
            while (true) {
                skip_ws();
                value.push_back(read_number<T>());
                skip_ws();
                if (peek() == ',') { get(); continue; }
                if (peek() == ']') break;
                throw std::runtime_error("expected ',' or ']'");
            }
        }
        expect(']');
        return true;
    }

    // --- Bulk data ---

    template<typename T>
        requires std::is_arithmetic_v<T>
    auto read_data(T* ptr, std::size_t count) -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
            expect('=');
        }
        expect('[');
        for (std::size_t i = 0; i < count; ++i) {
            skip_ws();
            ptr[i] = read_number<T>();
            skip_ws();
            if (i < count - 1) expect(',');
        }
        skip_ws();
        expect(']');
        return true;
    }

    // Overload for vec_t elements: read as flattened scalar array
    template<typename T, std::size_t N>
        requires std::is_arithmetic_v<T>
    auto read_data(vec_t<T, N>* ptr, std::size_t count) -> bool {
        return read_data(reinterpret_cast<T*>(ptr), count * N);
    }

    // --- CachedNdArray ---

    template<CachedNdArray T>
    auto read(T& arr) -> bool {
        if (!begin_group()) return false;
        using start_t = std::decay_t<decltype(start(arr))>;
        using shape_t = std::decay_t<decltype(shape(arr))>;
        start_t st;
        shape_t sh;
        begin_named("start");
        read(st);
        begin_named("shape");
        read(sh);
        arr = T(index_space(st, sh), memory::host);
        begin_named("data");
        read_data(data(arr), size(arr));
        end_group();
        return true;
    }

    // --- Groups ---

    auto begin_group() -> bool {
        if (pending_name_) {
            if (!seek_field(pending_name_)) {
                pending_name_ = nullptr;
                return false;
            }
            pending_name_ = nullptr;
        } else {
            skip_ws();
            if (peek() != '{') return false;
        }
        expect('{');
        group_stack_.push_back(is_.tellg());
        return true;
    }

    void end_group() {
        skip_to_group_end();
        expect('}');
        if (!group_stack_.empty()) {
            group_stack_.pop_back();
        }
    }

    auto begin_list() -> bool { return begin_group(); }
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
        expect('{');

        std::size_t count = 0;
        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();
            if (c == '{') {
                get();
                if (depth == 0) count++;
                depth++;
            } else if (c == '}') {
                if (depth == 0) break;
                get();
                depth--;
            } else if (c == std::char_traits<char>::eof()) {
                break;
            } else {
                get();
            }
        }
        is_.seekg(pos);
        return count;
    }

    auto count_strings(const char* name) -> std::size_t {
        auto pos = is_.tellg();
        if (!seek_field(name)) {
            is_.seekg(pos);
            return 0;
        }
        expect('{');

        std::size_t count = 0;
        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();
            if (c == '}') {
                if (depth == 0) break;
                get();
                depth--;
            } else if (c == '{') {
                get();
                depth++;
            } else if (c == '"' && depth == 0) {
                count++;
                read_quoted_string();
            } else if (c == std::char_traits<char>::eof()) {
                break;
            } else {
                get();
            }
        }
        is_.seekg(pos);
        return count;
    }

private:
    std::istream& is_;
    std::vector<std::streampos> group_stack_;
    const char* pending_name_ = nullptr;

    auto peek() -> char { return static_cast<char>(is_.peek()); }
    auto get() -> char { return static_cast<char>(is_.get()); }

    void skip_ws() {
        while (is_) {
            while (is_ && std::isspace(peek())) get();
            if (peek() == '#') {
                while (is_ && get() != '\n') {}
            } else {
                break;
            }
        }
    }

    void expect(char c) {
        skip_ws();
        if (get() != c) {
            throw std::runtime_error(std::string("expected '") + c + "'");
        }
    }

    auto read_identifier() -> std::string {
        std::string s;
        while (is_ && (std::isalnum(peek()) || peek() == '_')) {
            s += get();
        }
        return s;
    }

    template<typename T>
    auto read_number() -> T {
        skip_ws();
        std::string token;
        while (is_) {
            char c = peek();
            if (std::isdigit(c) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E') {
                token += get();
            } else {
                break;
            }
        }
        T value;
        std::istringstream iss(token);
        iss >> value;
        if (iss.fail()) {
            throw std::runtime_error("failed to parse number: " + token);
        }
        return value;
    }

    auto read_quoted_string() -> std::string {
        skip_ws();
        expect('"');
        std::string result;
        while (is_) {
            char c = get();
            if (c == '"') break;
            if (c == '\\') {
                char next = get();
                switch (next) {
                    case '\\': result += '\\'; break;
                    case '"':  result += '"'; break;
                    case 'n':  result += '\n'; break;
                    case 't':  result += '\t'; break;
                    case 'r':  result += '\r'; break;
                    default:   result += next; break;
                }
            } else {
                result += c;
            }
        }
        return result;
    }

    // Seek to a field by name within the current group
    auto seek_field(const char* name) -> bool {
        auto start = group_stack_.empty() ? std::streampos(0) : group_stack_.back();
        is_.seekg(start);

        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();

            if (c == '}') {
                if (depth == 0) return false;  // End of current group
                get();
                depth--;
            } else if (c == '{') {
                get();
                depth++;
            } else if (c == std::char_traits<char>::eof()) {
                return false;
            } else if (depth == 0 && (std::isalpha(c) || c == '_')) {
                auto id = read_identifier();
                if (id == name) {
                    skip_ws();
                    return true;
                }
                // Skip this field's value
                skip_field_value();
            } else {
                get();
            }
        }
        return false;
    }

    void skip_field_value() {
        skip_ws();
        char c = peek();
        if (c == '=') {
            get();
            skip_ws();
            c = peek();
            if (c == '"') {
                read_quoted_string();
            } else if (c == '[') {
                skip_bracketed();
            } else {
                // Skip scalar value
                while (is_ && !std::isspace(peek()) && peek() != '}' && peek() != '{') {
                    get();
                }
            }
        } else if (c == '{') {
            skip_braced();
        }
    }

    void skip_bracketed() {
        expect('[');
        int depth = 1;
        while (is_ && depth > 0) {
            char c = get();
            if (c == '[') depth++;
            else if (c == ']') depth--;
        }
    }

    void skip_braced() {
        expect('{');
        int depth = 1;
        while (is_ && depth > 0) {
            char c = get();
            if (c == '{') depth++;
            else if (c == '}') depth--;
        }
    }

    void skip_to_group_end() {
        int depth = 0;
        while (is_) {
            skip_ws();
            char c = peek();
            if (c == '{') {
                get();
                depth++;
            } else if (c == '}') {
                if (depth == 0) return;
                get();
                depth--;
            } else if (c == std::char_traits<char>::eof()) {
                return;
            } else {
                get();
            }
        }
    }
};

} // namespace mist
