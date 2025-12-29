#pragma once

// Source implementations for the parallel I/O serialization protocol.
// Provides ascii_source and binary_source classes.

#include <cctype>
#include <cstdint>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "format.hpp"

namespace archive {

// ============================================================================
// ascii_source - key-based lookup, missing fields return false
// ============================================================================

class ascii_source {
public:
    explicit ascii_source(std::istream& stream) : is(stream) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name = name;
    }

    // --- Scalars ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    auto read(T& value) -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
            expect('=');
        }
        value = read_number<T>();
        return true;
    }

    auto read(std::string& value) -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
            expect('=');
        } else {
            skip_ws();
            if (peek() != '"') return false;
        }
        value = read_quoted_string();
        return true;
    }

    // --- Arrays ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    auto read(std::vector<T>& value) -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
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

    // --- Groups ---

    auto begin_group() -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
        } else {
            skip_ws();
            if (peek() != '{') return false;
        }
        expect('{');
        group_stack.push_back(is.tellg());
        return true;
    }

    void end_group() {
        skip_to_group_end();
        expect('}');
        if (!group_stack.empty()) {
            group_stack.pop_back();
        }
    }

    auto begin_list() -> bool { return begin_group(); }
    void end_list() { end_group(); }

    // --- Query ---

    auto has_field(const char* name) -> bool {
        auto pos = is.tellg();
        bool found = seek_field(name);
        is.seekg(pos);
        return found;
    }

    auto count_items(const char* name) -> std::size_t {
        auto pos = is.tellg();
        if (!seek_field(name)) {
            is.seekg(pos);
            return 0;
        }
        expect('{');

        std::size_t count = 0;
        int depth = 0;
        while (is) {
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
        is.seekg(pos);
        return count;
    }

private:
    std::istream& is;
    std::vector<std::streampos> group_stack;
    const char* pending_name = nullptr;

    auto peek() -> char { return static_cast<char>(is.peek()); }
    auto get() -> char { return static_cast<char>(is.get()); }

    void skip_ws() {
        while (is) {
            while (is && std::isspace(peek())) get();
            if (peek() == '#') {
                while (is && get() != '\n') {}
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
        skip_ws();
        std::string s;
        while (is && (std::isalnum(peek()) || peek() == '_')) {
            s += get();
        }
        return s;
    }

    template <typename T>
    auto read_number() -> T {
        skip_ws();
        std::string token;
        while (is) {
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
        while (is) {
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

    auto seek_field(const char* name) -> bool {
        auto start = group_stack.empty() ? std::streampos(0) : group_stack.back();
        is.seekg(start);

        int depth = 0;
        while (is) {
            skip_ws();
            char c = peek();

            if (c == '}') {
                if (depth == 0) return false;
                get();
                depth--;
            } else if (c == '{') {
                get();
                depth++;
            } else if (c == std::char_traits<char>::eof()) {
                return false;
            } else if (depth == 0 && (std::isalnum(c) || c == '_')) {
                auto id = read_identifier();
                if (id == name) {
                    skip_ws();
                    return true;
                }
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
                while (is && !std::isspace(peek()) && peek() != '}' && peek() != '{') {
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
        while (is && depth > 0) {
            char c = get();
            if (c == '[') depth++;
            else if (c == ']') depth--;
        }
    }

    void skip_braced() {
        expect('{');
        int depth = 1;
        while (is && depth > 0) {
            char c = get();
            if (c == '{') depth++;
            else if (c == '}') depth--;
        }
    }

    void skip_to_group_end() {
        int depth = 0;
        while (is) {
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

// ============================================================================
// binary_source - key-based lookup, missing fields return false
// ============================================================================

class binary_source {
public:
    explicit binary_source(std::istream& stream, bool skip_header = false)
        : is(stream), header_read(skip_header), base_position(stream.tellg()) {}

    // --- Name context ---

    void begin_named(const char* name) {
        pending_name = name;
    }

    // --- Scalars ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    auto read(T& value) -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
        } else if (!check_list_item()) {
            return false;
        }
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
            throw std::runtime_error("Expected scalar type");
        }
        return true;
    }

    auto read(std::string& value) -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
        } else if (!check_list_item()) {
            return false;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_STRING) {
            throw std::runtime_error("Expected string type");
        }
        value = read_string_data();
        return true;
    }

    // --- Arrays ---

    template <typename T>
        requires std::is_arithmetic_v<T>
    auto read(std::vector<T>& value) -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_ARRAY) {
            throw std::runtime_error("Expected array type");
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

    // --- Groups ---

    auto begin_group() -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
        } else if (!group_stack.empty() && group_stack.back().is_list) {
            if (group_stack.back().remaining == 0) {
                return false;
            }
            group_stack.back().remaining--;
        } else if (group_stack.empty()) {
            ensure_header();
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_GROUP) {
            return false;
        }
        uint64_t field_count;
        read_raw(field_count);
        group_stack.push_back({is.tellg(), field_count, false});
        return true;
    }

    void end_group() {
        if (!group_stack.empty()) {
            group_stack.pop_back();
        }
    }

    auto begin_list() -> bool {
        if (pending_name) {
            if (!seek_field(pending_name)) {
                pending_name = nullptr;
                return false;
            }
            pending_name = nullptr;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_LIST) {
            return false;
        }
        uint64_t item_count;
        read_raw(item_count);
        group_stack.push_back({is.tellg(), item_count, true});
        return true;
    }

    void end_list() { end_group(); }

    // --- Query ---

    auto has_field(const char* name) -> bool {
        auto pos = is.tellg();
        bool found = seek_field(name);
        is.seekg(pos);
        return found;
    }

    auto count_items(const char* name) -> std::size_t {
        auto pos = is.tellg();
        if (!seek_field(name)) {
            is.seekg(pos);
            return 0;
        }
        uint8_t type_tag = read_type_tag();
        if (type_tag != binary_format::TYPE_LIST && type_tag != binary_format::TYPE_GROUP) {
            is.seekg(pos);
            return 0;
        }
        uint64_t count;
        read_raw(count);
        is.seekg(pos);
        return static_cast<std::size_t>(count);
    }

private:
    std::istream& is;
    const char* pending_name = nullptr;

    struct group_info {
        std::streampos start;
        uint64_t remaining;
        bool is_list;
    };
    std::vector<group_info> group_stack;
    bool header_read = false;
    std::streampos base_position = 0;

    auto check_list_item() -> bool {
        if (!group_stack.empty() && group_stack.back().is_list) {
            if (group_stack.back().remaining == 0) {
                return false;
            }
            group_stack.back().remaining--;
        }
        return true;
    }

    void ensure_header() {
        if (header_read) return;
        is.seekg(base_position);
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
        header_read = true;
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
            is.read(name.data(), static_cast<std::streamsize>(length));
            if (!is) {
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
            is.read(value.data(), static_cast<std::streamsize>(length));
            if (!is) {
                throw std::runtime_error("Failed to read string data");
            }
        }
        return value;
    }

    template <typename T>
    void read_raw(T& value) {
        is.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!is) {
            throw std::runtime_error("Failed to read data from binary archive");
        }
    }

    template <typename T>
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
            is.seekg(len, std::ios::cur);
        } else if (type_tag == binary_format::TYPE_ARRAY) {
            uint8_t elem_tag;
            read_raw(elem_tag);
            uint64_t count;
            read_raw(count);
            std::size_t elem_size = (elem_tag == binary_format::ELEM_INT32) ? 4 : 8;
            is.seekg(count * elem_size, std::ios::cur);
        } else if (type_tag == binary_format::TYPE_GROUP) {
            uint64_t field_count;
            read_raw(field_count);
            for (uint64_t i = 0; i < field_count; ++i) {
                read_name();
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

    auto seek_field(const char* name) -> bool {
        ensure_header();

        auto start = group_stack.empty() ? base_position + std::streamoff(5) : group_stack.back().start;
        is.seekg(start);

        if (!group_stack.empty() && group_stack.back().is_list) {
            return false;
        }

        uint64_t fields_to_scan = group_stack.empty() ? UINT64_MAX : group_stack.back().remaining;

        for (uint64_t i = 0; i < fields_to_scan && is; ++i) {
            auto field_name = read_name();
            if (field_name.empty() && !is) {
                return false;
            }
            if (field_name == name) {
                return true;
            }
            uint8_t type_tag = read_type_tag();
            skip_field_value(type_tag);
        }
        return false;
    }
};

} // namespace archive
