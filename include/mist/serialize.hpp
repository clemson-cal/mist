#pragma once

#include <concepts>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include "core.hpp"

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

// =============================================================================
// Deserialize implementation (forward declarations)
// =============================================================================

template<ArchiveReader A, typename T>
    requires std::is_arithmetic_v<T>
void deserialize(A& ar, const char* name, T& value);

template<ArchiveReader A>
void deserialize(A& ar, const char* name, std::string& value);

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

} // namespace mist
