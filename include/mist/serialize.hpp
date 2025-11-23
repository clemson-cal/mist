#pragma once

#include <concepts>
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
    { ar.write_scalar(name, int{}) } -> std::same_as<void>;
    { ar.write_scalar(name, double{}) } -> std::same_as<void>;
    { ar.begin_group(name) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;
    { ar.end_group() } -> std::same_as<void>;
};

template<typename A>
concept ArchiveReader = requires(A& ar, const char* name, int& i, double& d) {
    { ar.read_scalar(name, i) } -> std::same_as<void>;
    { ar.read_scalar(name, d) } -> std::same_as<void>;
    { ar.begin_group(name) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;
    { ar.end_group() } -> std::same_as<void>;
    { ar.count_groups(name) } -> std::same_as<std::size_t>;
};

// =============================================================================
// Serialize implementation (forward declarations)
// =============================================================================

template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const char* name, const T& value);

template<ArchiveWriter A>
void serialize(A& ar, const char* name, const std::string& value);

template<ArchiveWriter A, typename E>
    requires HasEnumStrings<E>
void serialize(A& ar, const char* name, const E& value);

template<ArchiveWriter A, typename T, std::size_t N>
void serialize(A& ar, const char* name, const vec_t<T, N>& value);

template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const char* name, const std::vector<T>& value);

template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const char* name, const std::vector<T>& value);

template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const char* name, const T& value);

template<ArchiveWriter A, typename T, std::size_t S>
void serialize(A& ar, const char* name, const cached_t<T, S>& value);

// =============================================================================
// Deserialize implementation (forward declarations)
// =============================================================================

template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
void deserialize(A& ar, const char* name, T& value);

template<ArchiveReader A>
void deserialize(A& ar, const char* name, std::string& value);

template<ArchiveReader A, typename E>
    requires HasEnumStrings<E>
void deserialize(A& ar, const char* name, E& value);

template<ArchiveReader A, typename T, std::size_t N>
void deserialize(A& ar, const char* name, vec_t<T, N>& value);

template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
void deserialize(A& ar, const char* name, std::vector<T>& value);

template<ArchiveReader A, typename T>
    requires HasFields<T>
void deserialize(A& ar, const char* name, std::vector<T>& value);

template<ArchiveReader A, typename T>
    requires HasFields<T>
void deserialize(A& ar, const char* name, T& value);

template<ArchiveReader A, typename T, std::size_t S>
void deserialize(A& ar, const char* name, cached_t<T, S>& value);

// =============================================================================
// Serialize implementations
// =============================================================================

// Scalar types
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const char* name, const T& value) {
    ar.write_scalar(name, value);
}

// std::string
template<ArchiveWriter A>
void serialize(A& ar, const char* name, const std::string& value) {
    ar.write_string(name, value);
}

// Enums with ADL to_string/from_string
template<ArchiveWriter A, typename E>
    requires HasEnumStrings<E>
void serialize(A& ar, const char* name, const E& value) {
    ar.write_string(name, to_string(value));
}

// vec_t<T, N>
template<ArchiveWriter A, typename T, std::size_t N>
void serialize(A& ar, const char* name, const vec_t<T, N>& value) {
    ar.write_array(name, value);
}

// std::vector<T> where T is arithmetic
template<ArchiveWriter A, typename T>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const char* name, const std::vector<T>& value) {
    ar.write_array(name, value);
}

// std::vector<T> where T is a compound type
template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const char* name, const std::vector<T>& value) {
    ar.begin_list(name);
    for (const auto& elem : value) {
        ar.begin_group();
        std::apply([&ar](auto&&... fields) {
            (serialize(ar, fields.name, fields.value), ...);
        }, elem.fields());
        ar.end_group();
    }
    ar.end_list();
}

// Compound types with fields()
template<ArchiveWriter A, typename T>
    requires HasConstFields<T>
void serialize(A& ar, const char* name, const T& value) {
    ar.begin_group(name);
    std::apply([&ar](auto&&... fields) {
        (serialize(ar, fields.name, fields.value), ...);
    }, value.fields());
    ar.end_group();
}

// cached_t<T, S> (ndarray)
template<ArchiveWriter A, typename T, std::size_t S>
void serialize(A& ar, const char* name, const cached_t<T, S>& arr) {
    static_assert(std::is_arithmetic_v<T>, "cached_t serialization requires arithmetic element type");
    if (location(arr) != memory::host) {
        throw std::runtime_error("cached_t serialization requires host memory");
    }
    ar.begin_group(name);
    serialize(ar, "start", start(arr));
    serialize(ar, "shape", shape(arr));
    ar.write_data("data", data(arr), size(arr));
    ar.end_group();
}

// =============================================================================
// Deserialize implementations
// =============================================================================

// Scalar types
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
void deserialize(A& ar, const char* name, T& value) {
    ar.read_scalar(name, value);
}

// std::string
template<ArchiveReader A>
void deserialize(A& ar, const char* name, std::string& value) {
    ar.read_string(name, value);
}

// Enums with ADL to_string/from_string
template<ArchiveReader A, typename E>
    requires HasEnumStrings<E>
void deserialize(A& ar, const char* name, E& value) {
    std::string str;
    ar.read_string(name, str);
    value = from_string(std::type_identity<E>{}, str);
}

// vec_t<T, N>
template<ArchiveReader A, typename T, std::size_t N>
void deserialize(A& ar, const char* name, vec_t<T, N>& value) {
    ar.read_array(name, value);
}

// std::vector<T> where T is arithmetic
template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
void deserialize(A& ar, const char* name, std::vector<T>& value) {
    ar.read_array(name, value);
}

// std::vector<T> where T is a compound type
template<ArchiveReader A, typename T>
    requires HasFields<T>
void deserialize(A& ar, const char* name, std::vector<T>& value) {
    std::size_t count = ar.count_groups(name);
    ar.begin_list(name);
    value.resize(count);
    for (auto& elem : value) {
        ar.begin_group();
        std::apply([&ar](auto&&... fields) {
            (deserialize(ar, fields.name, fields.value), ...);
        }, elem.fields());
        ar.end_group();
    }
    ar.end_list();
}

// Compound types with fields()
template<ArchiveReader A, typename T>
    requires HasFields<T>
void deserialize(A& ar, const char* name, T& value) {
    ar.begin_group(name);
    std::apply([&ar](auto&&... fields) {
        (deserialize(ar, fields.name, fields.value), ...);
    }, value.fields());
    ar.end_group();
}

// cached_t<T, S> (ndarray)
template<ArchiveReader A, typename T, std::size_t S>
void deserialize(A& ar, const char* name, cached_t<T, S>& arr) {
    static_assert(std::is_arithmetic_v<T>, "cached_t deserialization requires arithmetic element type");
    ar.begin_group(name);
    ivec_t<S> st;
    uvec_t<S> sh;
    deserialize(ar, "start", st);
    deserialize(ar, "shape", sh);
    arr = cached_t<T, S>(index_space(st, sh), memory::host);
    ar.read_data("data", data(arr), size(arr));
    ar.end_group();
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
