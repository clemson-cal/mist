#pragma once

// =============================================================================
// Mist Serialization - extends serialize library with mist types
// =============================================================================

#include "serialize/serialize.hpp"
#include "core.hpp"
#include "ndarray.hpp"

namespace mist {

// =============================================================================
// Re-export serialize:: symbols into mist namespace
// =============================================================================

using serialize::field;
using serialize::HasFields;
using serialize::HasConstFields;
using serialize::HasEnumStrings;
using serialize::ArchiveWriter;
using serialize::ArchiveReader;
using serialize::serialize;
using serialize::deserialize;
using serialize::set;

// Archive classes
using serialize::ascii_writer;
using serialize::ascii_reader;
using serialize::binary_writer;
using serialize::binary_reader;

// Parallel IO
using serialize::ParallelWrite;
using serialize::ParallelRead;
using serialize::HasItemKey;
using serialize::parallel_write;
using serialize::parallel_read;
using serialize::write_items;
using serialize::read_items;
using serialize::list_item_keys;
using serialize::read_header;

// =============================================================================
// Type traits for mist-specific types
// =============================================================================

// Check if type is a vec_t
template<typename T>
struct is_vec : std::false_type {};

template<typename T, std::size_t N>
struct is_vec<vec_t<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

// =============================================================================
// vec_t<T, N> serialization
// =============================================================================

template<ArchiveWriter A, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const vec_t<T, N>& value) {
    ar.write_data(&value[0], N);
}

template<ArchiveWriter A, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
void serialize(A& ar, const char* name, const vec_t<T, N>& value) {
    ar.begin_named(name);
    serialize(ar, value);
}

template<ArchiveReader A, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, vec_t<T, N>& value) -> bool {
    return ar.read_data(&value[0], N);
}

template<ArchiveReader A, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
auto deserialize(A& ar, const char* name, vec_t<T, N>& value) -> bool {
    ar.begin_named(name);
    return deserialize(ar, value);
}

// =============================================================================
// CachedNdArray serialization
// =============================================================================

template<ArchiveWriter A, CachedNdArray T>
void serialize(A& ar, const T& arr) {
    using value_type = typename T::value_type;

    ar.begin_group();
    mist::serialize(ar, "start", start(arr));
    mist::serialize(ar, "shape", shape(arr));
    ar.begin_named("data");
    if constexpr (is_vec_v<value_type>) {
        // Flatten vec_t elements to scalar array
        constexpr auto N = value_type::extent;
        using scalar_type = typename value_type::value_type;
        ar.write_data(reinterpret_cast<const scalar_type*>(data(arr)), size(arr) * N);
    } else {
        ar.write_data(data(arr), size(arr));
    }
    ar.end_group();
}

template<ArchiveWriter A, CachedNdArray T>
void serialize(A& ar, const char* name, const T& value) {
    ar.begin_named(name);
    serialize(ar, value);
}

template<ArchiveReader A, CachedNdArray T>
auto deserialize(A& ar, T& arr) -> bool {
    using value_type = typename T::value_type;
    constexpr auto D = T::rank;

    if (!ar.begin_group()) return false;
    auto s = ivec_t<D>{};
    auto n = uvec_t<D>{};
    mist::deserialize(ar, "start", s);
    mist::deserialize(ar, "shape", n);
    arr = T(index_space(s, n), memory::host);
    ar.begin_named("data");
    if constexpr (is_vec_v<value_type>) {
        constexpr auto N = value_type::extent;
        using scalar_type = typename value_type::value_type;
        ar.read_data(reinterpret_cast<scalar_type*>(data(arr)), size(arr) * N);
    } else {
        ar.read_data(data(arr), size(arr));
    }
    ar.end_group();
    return true;
}

template<ArchiveReader A, CachedNdArray T>
auto deserialize(A& ar, const char* name, T& value) -> bool {
    ar.begin_named(name);
    return deserialize(ar, value);
}

} // namespace mist
