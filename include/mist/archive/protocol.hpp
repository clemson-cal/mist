#pragma once

// I/O functions for the parallel I/O serialization protocol.
// Provides write/read free functions for common types.

#include <array>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "sink.hpp"
#include "source.hpp"

namespace archive {

// ============================================================================
// Field helper - returns std::pair<const char*, T&>
// ============================================================================

template <typename T>
constexpr auto field(const char* name, T& value) {
    return std::pair<const char*, T&>{name, value};
}

template <typename T>
constexpr auto field(const char* name, const T& value) {
    return std::pair<const char*, const T&>{name, value};
}

// ============================================================================
// HasFields concept - requires ADL free function fields(t)
// ============================================================================

template <typename T>
concept HasFields = requires(T& t) {
    { fields(t) };
};

template <typename T>
concept HasConstFields = requires(const T& t) {
    { fields(t) };
};

// ============================================================================
// Enum string conversion via ADL
// ============================================================================

template <typename E>
concept HasEnumStrings = std::is_enum_v<E> && requires(E e, const std::string& s) {
    { to_string(e) } -> std::convertible_to<const char*>;
    { from_string(std::type_identity<E>{}, s) } -> std::same_as<E>;
};

// ============================================================================
// Write declarations (two-arg: anonymous, three-arg: named)
// ============================================================================

// Named write - convenience wrapper
template <typename Sink, typename T>
void write(Sink& sink, const char* name, const T& value);

// Anonymous write - core implementations
template <typename Sink, typename T>
    requires std::is_arithmetic_v<T>
void write(Sink& sink, const T& value);

template <typename Sink>
void write(Sink& sink, const std::string& value);

template <typename Sink, typename E>
    requires HasEnumStrings<E>
void write(Sink& sink, const E& value);

template <typename Sink, typename T1, typename T2>
void write(Sink& sink, const std::pair<T1, T2>& value);

template <typename Sink, typename T>
    requires std::is_arithmetic_v<T>
void write(Sink& sink, const std::vector<T>& value);

template <typename Sink, typename T>
    requires (!std::is_arithmetic_v<T>)
void write(Sink& sink, const std::vector<T>& value);

template <typename Sink, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
void write(Sink& sink, const std::array<T, N>& value);

template <typename Sink, typename T, std::size_t N>
    requires (!std::is_arithmetic_v<T>)
void write(Sink& sink, const std::array<T, N>& value);

template <typename Sink, typename T>
void write(Sink& sink, const std::map<std::string, T>& value);

template <typename Sink, typename T>
    requires HasConstFields<T>
void write(Sink& sink, const T& value);

template <typename Sink, typename T>
void write(Sink& sink, const std::optional<T>& value);

template <typename Sink, typename... Ts>
void write(Sink& sink, const std::variant<Ts...>& value);

// ============================================================================
// Read declarations (two-arg: anonymous, three-arg: named)
// ============================================================================

// Named read - convenience wrapper
template <typename Source, typename T>
auto read(Source& source, const char* name, T& value) -> bool;

// Anonymous read - core implementations
template <typename Source, typename T>
    requires std::is_arithmetic_v<T>
auto read(Source& source, T& value) -> bool;

template <typename Source>
auto read(Source& source, std::string& value) -> bool;

template <typename Source, typename E>
    requires HasEnumStrings<E>
auto read(Source& source, E& value) -> bool;

template <typename Source, typename T1, typename T2>
auto read(Source& source, std::pair<T1, T2>& value) -> bool;

template <typename Source, typename T>
    requires std::is_arithmetic_v<T>
auto read(Source& source, std::vector<T>& value) -> bool;

template <typename Source, typename T>
    requires (!std::is_arithmetic_v<T>)
auto read(Source& source, std::vector<T>& value) -> bool;

template <typename Source, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
auto read(Source& source, std::array<T, N>& value) -> bool;

template <typename Source, typename T, std::size_t N>
    requires (!std::is_arithmetic_v<T>)
auto read(Source& source, std::array<T, N>& value) -> bool;

template <typename Source, typename T>
auto read(Source& source, std::map<std::string, T>& value) -> bool;

template <typename Source, typename T>
    requires HasFields<T>
auto read(Source& source, T& value) -> bool;

template <typename Source, typename T>
auto read(Source& source, std::optional<T>& value) -> bool;

template <typename Source, typename... Ts>
auto read(Source& source, std::variant<Ts...>& value) -> bool;

// ============================================================================
// Named wrappers (three-arg versions)
// ============================================================================

template <typename Sink, typename T>
void write(Sink& sink, const char* name, const T& value) {
    sink.begin_named(name);
    write(sink, value);
}

template <typename Source, typename T>
auto read(Source& source, const char* name, T& value) -> bool {
    source.begin_named(name);
    return read(source, value);
}

// ============================================================================
// Write implementations (two-arg anonymous versions)
// ============================================================================

// Scalar types
template <typename Sink, typename T>
    requires std::is_arithmetic_v<T>
void write(Sink& sink, const T& value) {
    sink.write(value);
}

// std::string
template <typename Sink>
void write(Sink& sink, const std::string& value) {
    sink.write(value);
}

// Enums with ADL to_string/from_string
template <typename Sink, typename E>
    requires HasEnumStrings<E>
void write(Sink& sink, const E& value) {
    sink.write(std::string(to_string(value)));
}

// std::pair<T1, T2>
template <typename Sink, typename T1, typename T2>
void write(Sink& sink, const std::pair<T1, T2>& value) {
    sink.begin_group();
    write(sink, "first", value.first);
    write(sink, "second", value.second);
    sink.end_group();
}

// std::vector<T> where T is arithmetic - use sink.write() directly
template <typename Sink, typename T>
    requires std::is_arithmetic_v<T>
void write(Sink& sink, const std::vector<T>& value) {
    sink.write(value);
}

// std::vector<T> where T is not arithmetic - write each element
template <typename Sink, typename T>
    requires (!std::is_arithmetic_v<T>)
void write(Sink& sink, const std::vector<T>& value) {
    sink.begin_list();
    for (const auto& elem : value) {
        write(sink, elem);
    }
    sink.end_list();
}

// std::array<T, N> where T is arithmetic - convert to vector
template <typename Sink, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
void write(Sink& sink, const std::array<T, N>& value) {
    sink.write(std::vector<T>(value.begin(), value.end()));
}

// std::array<T, N> where T is not arithmetic - write each element
template <typename Sink, typename T, std::size_t N>
    requires (!std::is_arithmetic_v<T>)
void write(Sink& sink, const std::array<T, N>& value) {
    sink.begin_list();
    for (const auto& elem : value) {
        write(sink, elem);
    }
    sink.end_list();
}

// std::map<std::string, T>
template <typename Sink, typename T>
void write(Sink& sink, const std::map<std::string, T>& value) {
    sink.begin_list();
    for (const auto& [key, val] : value) {
        sink.begin_group();
        write(sink, "key", key);
        write(sink, "value", val);
        sink.end_group();
    }
    sink.end_list();
}

// Compound types with fields()
template <typename Sink, typename T>
    requires HasConstFields<T>
void write(Sink& sink, const T& value) {
    sink.begin_group();
    std::apply([&sink](auto&&... f) {
        (write(sink, f.first, f.second), ...);
    }, fields(value));
    sink.end_group();
}

// std::optional<T>
template <typename Sink, typename T>
void write(Sink& sink, const std::optional<T>& value) {
    sink.begin_group();
    bool has_value = value.has_value();
    write(sink, "has_value", has_value);
    if (has_value) {
        write(sink, "value", *value);
    }
    sink.end_group();
}

// std::variant<Ts...>
template <typename Sink, typename... Ts>
void write(Sink& sink, const std::variant<Ts...>& value) {
    sink.begin_group();
    write(sink, "index", value.index());
    std::visit([&sink](const auto& v) { write(sink, "value", v); }, value);
    sink.end_group();
}

// ============================================================================
// Read implementations (two-arg anonymous versions)
// ============================================================================

// Scalar types
template <typename Source, typename T>
    requires std::is_arithmetic_v<T>
auto read(Source& source, T& value) -> bool {
    return source.read(value);
}

// std::string
template <typename Source>
auto read(Source& source, std::string& value) -> bool {
    return source.read(value);
}

// Enums with ADL to_string/from_string
template <typename Source, typename E>
    requires HasEnumStrings<E>
auto read(Source& source, E& value) -> bool {
    std::string str;
    if (!source.read(str)) return false;
    value = from_string(std::type_identity<E>{}, str);
    return true;
}

// std::pair<T1, T2>
template <typename Source, typename T1, typename T2>
auto read(Source& source, std::pair<T1, T2>& value) -> bool {
    if (!source.begin_group()) return false;
    read(source, "first", value.first);
    read(source, "second", value.second);
    source.end_group();
    return true;
}

// std::vector<T> where T is arithmetic - use source.read() directly
template <typename Source, typename T>
    requires std::is_arithmetic_v<T>
auto read(Source& source, std::vector<T>& value) -> bool {
    return source.read(value);
}

// std::vector<T> where T is not arithmetic - read each element
template <typename Source, typename T>
    requires (!std::is_arithmetic_v<T>)
auto read(Source& source, std::vector<T>& value) -> bool {
    if (!source.begin_list()) return false;
    value.clear();
    while (true) {
        T elem;
        if (!read(source, elem)) break;
        value.push_back(std::move(elem));
    }
    source.end_list();
    return true;
}

// std::array<T, N> where T is arithmetic - read via vector
template <typename Source, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
auto read(Source& source, std::array<T, N>& value) -> bool {
    std::vector<T> vec;
    if (!source.read(vec)) return false;
    if (vec.size() != N) return false;
    std::copy(vec.begin(), vec.end(), value.begin());
    return true;
}

// std::array<T, N> where T is not arithmetic - read each element
template <typename Source, typename T, std::size_t N>
    requires (!std::is_arithmetic_v<T>)
auto read(Source& source, std::array<T, N>& value) -> bool {
    if (!source.begin_list()) return false;
    for (std::size_t i = 0; i < N; ++i) {
        if (!read(source, value[i])) {
            source.end_list();
            return false;
        }
    }
    source.end_list();
    return true;
}

// std::map<std::string, T>
template <typename Source, typename T>
auto read(Source& source, std::map<std::string, T>& value) -> bool {
    if (!source.begin_list()) return false;
    value.clear();
    while (source.begin_group()) {
        std::string key;
        T val;
        read(source, "key", key);
        read(source, "value", val);
        value[key] = std::move(val);
        source.end_group();
    }
    source.end_list();
    return true;
}

// Compound types with fields()
template <typename Source, typename T>
    requires HasFields<T>
auto read(Source& source, T& value) -> bool {
    if (!source.begin_group()) return false;
    std::apply([&source](auto&&... f) {
        (read(source, f.first, f.second), ...);
    }, fields(value));
    source.end_group();
    return true;
}

// std::optional<T>
template <typename Source, typename T>
auto read(Source& source, std::optional<T>& value) -> bool {
    if (!source.begin_group()) return false;
    bool has_value = false;
    read(source, "has_value", has_value);
    if (has_value) {
        T temp;
        read(source, "value", temp);
        value = std::move(temp);
    } else {
        value = std::nullopt;
    }
    source.end_group();
    return true;
}

// std::variant<Ts...> helper
namespace detail {

template <typename Source, typename Variant, std::size_t I = 0>
auto read_variant_by_index(Source& source, Variant& value, std::size_t index) -> bool {
    if constexpr (I >= std::variant_size_v<Variant>) {
        return false;
    } else {
        if (I == index) {
            std::variant_alternative_t<I, Variant> temp;
            if (!read(source, temp)) return false;
            value = std::move(temp);
            return true;
        }
        return read_variant_by_index<Source, Variant, I + 1>(source, value, index);
    }
}

} // namespace detail

template <typename Source, typename... Ts>
auto read(Source& source, std::variant<Ts...>& value) -> bool {
    if (!source.begin_group()) return false;
    std::size_t index = 0;
    read(source, "index", index);
    source.begin_named("value");
    auto result = detail::read_variant_by_index(source, value, index);
    source.end_group();
    return result;
}

// ============================================================================
// Config field setter by path
// ============================================================================

namespace detail {

// Helper to parse string to target type
template <typename T>
void parse_and_assign(T& target, const std::string& value) {
    if constexpr (std::is_same_v<T, int>) {
        target = std::stoi(value);
    } else if constexpr (std::is_same_v<T, long>) {
        target = std::stol(value);
    } else if constexpr (std::is_same_v<T, long long>) {
        target = std::stoll(value);
    } else if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long>) {
        target = std::stoul(value);
    } else if constexpr (std::is_same_v<T, float>) {
        target = std::stof(value);
    } else if constexpr (std::is_same_v<T, double>) {
        target = std::stod(value);
    } else if constexpr (std::is_same_v<T, bool>) {
        target = (value == "true" || value == "1");
    } else if constexpr (std::is_same_v<T, std::string>) {
        target = value;
    } else if constexpr (HasEnumStrings<T>) {
        target = from_string(std::type_identity<T>{}, value);
    } else {
        throw std::runtime_error("unsupported type for set()");
    }
}

// Forward declaration
template <typename T>
void set_impl(T& obj, const std::string& path, const std::string& value);

// Set field on a leaf type (non-HasFields)
template <typename T>
void set_field(T& target, const std::string& rest, const std::string& value) {
    if (rest.empty()) {
        parse_and_assign(target, value);
    } else if constexpr (HasFields<T>) {
        set_impl(target, rest, value);
    } else {
        throw std::runtime_error("cannot descend into '" + rest + "': not a struct");
    }
}

// Helper to try setting a field if name matches
template <typename T>
void try_set_field(const char* name, T& target, const std::string& key,
                   const std::string& rest, const std::string& value, bool& found) {
    if (!found && std::string(name) == key) {
        set_field(target, rest, value);
        found = true;
    }
}

// Set field by path on a HasFields type
template <typename T>
void set_impl(T& obj, const std::string& path, const std::string& value) {
    auto dot = path.find('.');
    std::string key = path.substr(0, dot);
    std::string rest = (dot != std::string::npos) ? path.substr(dot + 1) : "";

    bool found = false;
    std::apply([&](auto&&... f) {
        (try_set_field(f.first, f.second, key, rest, value, found), ...);
    }, fields(obj));

    if (!found) {
        throw std::runtime_error("field not found: " + key);
    }
}

} // namespace detail

/**
 * Set a field in a struct by dot-separated path.
 *
 * Example:
 *   set(config, "physics.gamma", "1.33");
 *   set(config, "driver.t_final", "2.0");
 *   set(config, "mesh.boundary", "periodic");  // enum with to_string/from_string
 */
template <HasFields T>
void set(T& obj, const std::string& path, const std::string& value) {
    detail::set_impl(obj, path, value);
}

} // namespace archive
