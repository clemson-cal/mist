#pragma once

#include <concepts>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include "core.hpp"
#include "ndarray.hpp"

namespace mist {

// =============================================================================
// Field wrapper for named serialization
// =============================================================================

template<typename T>
struct field_t {
    const char* name;
    T& value;
};

template<typename T>
constexpr field_t<T> field(const char* name, T& value) {
    return field_t<T>{name, value};
}

template<typename T>
constexpr field_t<const T> field(const char* name, const T& value) {
    return field_t<const T>{name, value};
}

// =============================================================================
// Type traits for serialization
// =============================================================================

// Check if type is a vec_t
template<typename T>
struct is_vec : std::false_type {};

template<typename T, std::size_t N>
struct is_vec<vec_t<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

// Check if type is a std::vector
template<typename T>
struct is_std_vector : std::false_type {};

template<typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template<typename T>
inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

// Get element type of std::vector
template<typename T>
struct vector_element_type { using type = void; };

template<typename T, typename A>
struct vector_element_type<std::vector<T, A>> { using type = T; };

template<typename T>
using vector_element_type_t = typename vector_element_type<T>::type;

// =============================================================================
// Serializable concept
// =============================================================================

template<typename T>
concept HasFields = requires(T t) {
    { t.fields() } -> std::same_as<decltype(t.fields())>;
};

template<typename T>
concept HasConstFields = requires(const T t) {
    { t.fields() } -> std::same_as<decltype(t.fields())>;
};

// =============================================================================
// Enum string conversion via ADL
// =============================================================================

// Concept to detect if a type has ADL to_string/from_string for enum conversion
template<typename E>
concept HasEnumStrings = std::is_enum_v<E> && requires(E e, const std::string& s) {
    { to_string(e) } -> std::convertible_to<const char*>;
    { from_string(std::type_identity<E>{}, s) } -> std::same_as<E>;
};

// =============================================================================
// Archive concepts
// =============================================================================

template<typename A>
concept ArchiveWriter = requires(A& ar, const char* name) {
    { ar.begin_named(name) } -> std::same_as<void>;
    { ar.write(int{}) } -> std::same_as<void>;
    { ar.write(double{}) } -> std::same_as<void>;
    { ar.write(std::string{}) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;
    { ar.end_group() } -> std::same_as<void>;
    { ar.begin_list() } -> std::same_as<void>;
    { ar.end_list() } -> std::same_as<void>;
};

template<typename A>
concept ArchiveReader = requires(A& ar, const char* name, int& i, double& d, std::string& s) {
    { ar.begin_named(name) } -> std::same_as<void>;
    { ar.read(i) } -> std::same_as<bool>;
    { ar.read(d) } -> std::same_as<bool>;
    { ar.read(s) } -> std::same_as<bool>;
    { ar.begin_group() } -> std::same_as<bool>;
    { ar.end_group() } -> std::same_as<void>;
    { ar.begin_list() } -> std::same_as<bool>;
    { ar.end_list() } -> std::same_as<void>;
    { ar.has_field(name) } -> std::same_as<bool>;
    { ar.count_items(name) } -> std::same_as<std::size_t>;
};

// =============================================================================
// Serialize declarations (two-arg: anonymous, three-arg: named)
// =============================================================================

// Named serialization - convenience wrapper
template<ArchiveWriter A, typename T>
void serialize(A& ar, const char* name, const T& value);

// Anonymous serialization - core implementations
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const T& value);

template<ArchiveWriter A>
void serialize(A& ar, const std::string& value);

template<ArchiveWriter A, typename E>
    requires HasEnumStrings<E>
void serialize(A& ar, const E& value);

template<ArchiveWriter A, typename T1, typename T2>
void serialize(A& ar, const std::pair<T1, T2>& value);

template<ArchiveWriter A, typename T, std::size_t N>
void serialize(A& ar, const vec_t<T, N>& value);

template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const std::vector<T>& value);

template<ArchiveWriter A, typename T>
    requires (!std::is_arithmetic_v<T>)
void serialize(A& ar, const std::vector<T>& value);

template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::map<std::string, T>& value);

template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const T& value);

template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::optional<T>& value);

template<ArchiveWriter A, CachedNdArray T>
void serialize(A& ar, const T& value);

// =============================================================================
// Deserialize declarations (two-arg: anonymous, three-arg: named)
// =============================================================================

// Named deserialization - convenience wrapper
template<ArchiveReader A, typename T>
auto deserialize(A& ar, const char* name, T& value) -> bool;

// Anonymous deserialization - core implementations
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, T& value) -> bool;

template<ArchiveReader A>
auto deserialize(A& ar, std::string& value) -> bool;

template<ArchiveReader A, typename E>
    requires HasEnumStrings<E>
auto deserialize(A& ar, E& value) -> bool;

template<ArchiveReader A, typename T1, typename T2>
auto deserialize(A& ar, std::pair<T1, T2>& value) -> bool;

template<ArchiveReader A, typename T, std::size_t N>
auto deserialize(A& ar, vec_t<T, N>& value) -> bool;

template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, std::vector<T>& value) -> bool;

template<ArchiveReader A, typename T>
    requires (!std::is_arithmetic_v<T>)
auto deserialize(A& ar, std::vector<T>& value) -> bool;

template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::map<std::string, T>& value) -> bool;

template<ArchiveReader A, typename T>
    requires HasFields<T>
auto deserialize(A& ar, T& value) -> bool;

template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::optional<T>& value) -> bool;

template<ArchiveReader A, CachedNdArray T>
auto deserialize(A& ar, T& value) -> bool;

// =============================================================================
// Named wrappers (three-arg versions)
// =============================================================================

template<ArchiveWriter A, typename T>
void serialize(A& ar, const char* name, const T& value) {
    ar.begin_named(name);
    serialize(ar, value);
}

template<ArchiveReader A, typename T>
auto deserialize(A& ar, const char* name, T& value) -> bool {
    ar.begin_named(name);
    return deserialize(ar, value);
}

// =============================================================================
// Serialize implementations (two-arg anonymous versions)
// =============================================================================

// Scalar types
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const T& value) {
    ar.write(value);
}

// std::string
template<ArchiveWriter A>
void serialize(A& ar, const std::string& value) {
    ar.write(value);
}

// Enums with ADL to_string/from_string
template<ArchiveWriter A, typename E>
    requires HasEnumStrings<E>
void serialize(A& ar, const E& value) {
    ar.write(std::string(to_string(value)));
}

// std::pair<T1, T2>
template<ArchiveWriter A, typename T1, typename T2>
void serialize(A& ar, const std::pair<T1, T2>& value) {
    ar.begin_group();
    serialize(ar, "first", value.first);
    serialize(ar, "second", value.second);
    ar.end_group();
}

// vec_t<T, N>
template<ArchiveWriter A, typename T, std::size_t N>
void serialize(A& ar, const vec_t<T, N>& value) {
    ar.write(value);
}

// std::vector<T> where T is arithmetic - use ar.write() for efficiency
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const std::vector<T>& value) {
    ar.write(value);
}

// std::vector<T> where T is not arithmetic - serialize each element
template<ArchiveWriter A, typename T>
    requires (!std::is_arithmetic_v<T>)
void serialize(A& ar, const std::vector<T>& value) {
    ar.begin_list();
    for (const auto& elem : value) {
        serialize(ar, elem);
    }
    ar.end_list();
}

// std::map<std::string, T>
template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::map<std::string, T>& value) {
    ar.begin_list();
    for (const auto& [key, val] : value) {
        ar.begin_group();
        serialize(ar, "key", key);
        serialize(ar, "value", val);
        ar.end_group();
    }
    ar.end_list();
}

// Compound types with fields()
template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const T& value) {
    ar.begin_group();
    std::apply([&ar](auto&&... fields) {
        (serialize(ar, fields.name, fields.value), ...);
    }, value.fields());
    ar.end_group();
}

// std::optional<T>
template<ArchiveWriter A, typename T>
void serialize(A& ar, const std::optional<T>& value) {
    ar.begin_group();
    bool has_value = value.has_value();
    serialize(ar, "has_value", has_value);
    if (has_value) {
        serialize(ar, "value", *value);
    }
    ar.end_group();
}

// CachedNdArray - dispatches to ar.write()
template<ArchiveWriter A, CachedNdArray T>
void serialize(A& ar, const T& value) {
    ar.write(value);
}

// =============================================================================
// Deserialize implementations (two-arg anonymous versions)
// =============================================================================

// Scalar types
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, T& value) -> bool {
    return ar.read(value);
}

// std::string
template<ArchiveReader A>
auto deserialize(A& ar, std::string& value) -> bool {
    return ar.read(value);
}

// Enums with ADL to_string/from_string
template<ArchiveReader A, typename E>
    requires HasEnumStrings<E>
auto deserialize(A& ar, E& value) -> bool {
    std::string str;
    if (!ar.read(str)) return false;
    value = from_string(std::type_identity<E>{}, str);
    return true;
}

// std::pair<T1, T2>
template<ArchiveReader A, typename T1, typename T2>
auto deserialize(A& ar, std::pair<T1, T2>& value) -> bool {
    if (!ar.begin_group()) return false;
    deserialize(ar, "first", value.first);
    deserialize(ar, "second", value.second);
    ar.end_group();
    return true;
}

// vec_t<T, N>
template<ArchiveReader A, typename T, std::size_t N>
auto deserialize(A& ar, vec_t<T, N>& value) -> bool {
    return ar.read(value);
}

// std::vector<T> where T is arithmetic - use ar.read() for efficiency
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, std::vector<T>& value) -> bool {
    return ar.read(value);
}

// std::vector<T> where T is not arithmetic - deserialize each element
template<ArchiveReader A, typename T>
    requires (!std::is_arithmetic_v<T>)
auto deserialize(A& ar, std::vector<T>& value) -> bool {
    if (!ar.begin_list()) return false;
    value.clear();
    while (true) {
        T elem;
        if (!deserialize(ar, elem)) break;
        value.push_back(std::move(elem));
    }
    ar.end_list();
    return true;
}

// std::map<std::string, T>
template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::map<std::string, T>& value) -> bool {
    if (!ar.begin_list()) return false;
    value.clear();
    while (ar.begin_group()) {
        std::string key;
        T val;
        deserialize(ar, "key", key);
        deserialize(ar, "value", val);
        value[key] = std::move(val);
        ar.end_group();
    }
    ar.end_list();
    return true;
}

// Compound types with fields()
template<ArchiveReader A, typename T>
    requires HasFields<T>
auto deserialize(A& ar, T& value) -> bool {
    if (!ar.begin_group()) return false;
    std::apply([&ar](auto&&... fields) {
        (deserialize(ar, fields.name, fields.value), ...);
    }, value.fields());
    ar.end_group();
    return true;
}

// std::optional<T>
template<ArchiveReader A, typename T>
auto deserialize(A& ar, std::optional<T>& value) -> bool {
    if (!ar.begin_group()) return false;
    bool has_value = false;
    deserialize(ar, "has_value", has_value);
    if (has_value) {
        T temp;
        deserialize(ar, "value", temp);
        value = std::move(temp);
    } else {
        value = std::nullopt;
    }
    ar.end_group();
    return true;
}

// CachedNdArray - dispatches to ar.read()
template<ArchiveReader A, CachedNdArray T>
auto deserialize(A& ar, T& value) -> bool {
    return ar.read(value);
}

// =============================================================================
// Config field setter by path
// =============================================================================

namespace detail {

// Helper to parse string to target type
template<typename T>
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
template<typename T>
void set_impl(T& obj, const std::string& path, const std::string& value);

// Set field on a leaf type (non-HasFields)
template<typename T>
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
template<typename T>
void try_set_field(const char* name, T& target, const std::string& key,
                   const std::string& rest, const std::string& value, bool& found) {
    if (!found && std::string(name) == key) {
        set_field(target, rest, value);
        found = true;
    }
}

// Set field by path on a HasFields type
template<typename T>
void set_impl(T& obj, const std::string& path, const std::string& value) {
    auto dot = path.find('.');
    std::string key = path.substr(0, dot);
    std::string rest = (dot != std::string::npos) ? path.substr(dot + 1) : "";

    bool found = false;
    std::apply([&](auto&&... field) {
        (try_set_field(field.name, field.value, key, rest, value, found), ...);
    }, obj.fields());

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
template<HasFields T>
void set(T& obj, const std::string& path, const std::string& value) {
    detail::set_impl(obj, path, value);
}

} // namespace mist
