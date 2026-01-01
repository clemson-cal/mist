#pragma once

// =============================================================================
// Mist Archive - extends archive library with mist types
// =============================================================================

#include "archive/archive.hpp"
#include "core.hpp"
#include "ndarray.hpp"

namespace mist {

// =============================================================================
// Re-export archive:: symbols into mist namespace
// =============================================================================

using archive::field;
using archive::HasFields;
using archive::HasConstFields;
using archive::HasEnumStrings;
using archive::Sink;
using archive::Source;
using archive::write;
using archive::read;
using archive::set;

// Sink/Source classes
using archive::ascii_sink;
using archive::ascii_source;
using archive::binary_sink;
using archive::binary_source;

// Checkpoint concepts and functions
using archive::Checkpointable;
using archive::Emittable;
using archive::Gatherable;
using archive::PatchKey;
using archive::Scatterable;
using archive::Truthful;
using archive::Binary;
using archive::Ascii;
using archive::distributed_write;
using archive::distributed_read;

// =============================================================================
// Type traits for mist-specific types
// =============================================================================

template<typename T>
struct is_vec : std::false_type {};

template<typename T, std::size_t N>
struct is_vec<vec_t<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_vec_v = is_vec<T>::value;

// =============================================================================
// vec_t<T, N> serialization
// =============================================================================

template<Sink S, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
void write(S& sink, const vec_t<T, N>& value) {
    sink.write(std::vector<T>(&value[0], &value[0] + N));
}

template<Sink S, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
void write(S& sink, const char* name, const vec_t<T, N>& value) {
    sink.begin_named(name);
    write(sink, value);
}

template<Source S, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
auto read(S& source, vec_t<T, N>& value) -> bool {
    auto vec = std::vector<T>{};
    if (!source.read(vec)) return false;
    if (vec.size() != N) return false;
    std::copy(vec.begin(), vec.end(), &value[0]);
    return true;
}

template<Source S, typename T, std::size_t N>
    requires std::is_arithmetic_v<T>
auto read(S& source, const char* name, vec_t<T, N>& value) -> bool {
    source.begin_named(name);
    return read(source, value);
}

// =============================================================================
// CachedNdArray serialization
// =============================================================================

template<Sink S, CachedNdArray T>
void write(S& sink, const T& arr) {
    using value_type = typename T::value_type;
    constexpr auto D = T::rank;

    sink.begin_group();
    mist::write(sink, "start", start(arr));
    mist::write(sink, "shape", shape(arr));
    sink.begin_named("data");
    if constexpr (is_vec_v<value_type>) {
        constexpr auto N = value_type::extent;
        using scalar_type = typename value_type::value_type;
        auto vec = std::vector<scalar_type>(size(arr) * N);
        std::memcpy(vec.data(), data(arr), vec.size() * sizeof(scalar_type));
        sink.write(vec);
    } else {
        sink.write(std::vector<value_type>(data(arr), data(arr) + size(arr)));
    }
    sink.end_group();
}

template<Sink S, CachedNdArray T>
void write(S& sink, const char* name, const T& value) {
    sink.begin_named(name);
    write(sink, value);
}

template<Source S, CachedNdArray T>
auto read(S& source, T& arr) -> bool {
    using value_type = typename T::value_type;
    constexpr auto D = T::rank;

    if (!source.begin_group()) return false;
    auto s = ivec_t<D>{};
    auto n = uvec_t<D>{};
    mist::read(source, "start", s);
    mist::read(source, "shape", n);
    arr = T(index_space(s, n), memory::host);
    source.begin_named("data");
    if constexpr (is_vec_v<value_type>) {
        constexpr auto N = value_type::extent;
        using scalar_type = typename value_type::value_type;
        auto vec = std::vector<scalar_type>{};
        if (!source.read(vec)) return false;
        if (vec.size() != size(arr) * N) return false;
        std::memcpy(data(arr), vec.data(), vec.size() * sizeof(scalar_type));
    } else {
        auto vec = std::vector<value_type>{};
        if (!source.read(vec)) return false;
        if (vec.size() != size(arr)) return false;
        std::copy(vec.begin(), vec.end(), data(arr));
    }
    source.end_group();
    return true;
}

template<Source S, CachedNdArray T>
auto read(S& source, const char* name, T& value) -> bool {
    source.begin_named(name);
    return read(source, value);
}

// =============================================================================
// CachedNdArray patch key for parallel IO
// =============================================================================

template<CachedNdArray T>
auto to_string(const T& arr) -> std::string {
    auto s = start(arr);
    auto n = shape(arr);
    auto result = std::string{};
    for (std::size_t d = 0; d < T::rank; ++d) {
        if (d > 0) result += "_";
        result += std::to_string(s[d]) + "x" + std::to_string(n[d]);
    }
    return result;
}

} // namespace mist
