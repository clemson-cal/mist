#pragma once

// Parallel I/O Serialization Protocol
//
// Layers:
//   Layer 0 (existing):  Archive-style Sink/Source + Writable/Readable
//   Layer 1:             Truthful adapter for types with transient state
//   Layer 2:             Emittable/Scatterable/Gatherable for checkpoints
//
// Archive-style I/O uses key-based access (begin_named + write/read)
// for forward/backward compatibility with evolving data structures.

#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "sink.hpp"
#include "source.hpp"
#include "protocol.hpp"

namespace archive {

// ============================================================================
// Format tag types and enum
// ============================================================================

struct Binary {};
struct Ascii {};

enum class format {
    ascii,
    binary
};

// ============================================================================
// Format dispatch helpers
// ============================================================================

template <typename F>
auto with_sink(std::ostream& os, format fmt, F&& func) {
    if (fmt == format::binary) {
        auto sink = binary_sink{os};
        return func(sink);
    } else {
        auto sink = ascii_sink{os};
        return func(sink);
    }
}

template <typename F>
auto with_source(std::istream& is, format fmt, F&& func) {
    if (fmt == format::binary) {
        auto source = binary_source{is};
        return func(source);
    } else {
        auto source = ascii_source{is};
        return func(source);
    }
}

// ============================================================================
// Hashable concept (not in standard)
// ============================================================================

template <typename T>
concept Hashable = requires(const T& t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

// ============================================================================
// Sink concept (archive-style, key-based write-only)
// ============================================================================

template <typename S>
concept Sink = requires(S& s, const char* name, int i, double d, std::string str, std::vector<double> vec) {
    s.begin_named(name);      // set key for next write
    s.write(i);               // scalar int
    s.write(d);               // scalar double
    s.write(str);             // string
    s.write(vec);             // numeric array
    s.begin_group();          // open group/struct
    s.end_group();            // close group/struct
    s.begin_list();           // open list (anonymous items)
    s.end_list();             // close list
};

// ============================================================================
// Source concept (archive-style, key-based read-only)
// ============================================================================

template <typename S>
concept Source = requires(S& s, const char* name, int& i, double& d, std::string& str, std::vector<double>& vec) {
    s.begin_named(name);                          // set key for next read
    { s.read(i) } -> std::same_as<bool>;          // returns false if field missing
    { s.read(d) } -> std::same_as<bool>;
    { s.read(str) } -> std::same_as<bool>;
    { s.read(vec) } -> std::same_as<bool>;
    { s.begin_group() } -> std::same_as<bool>;    // false if group missing
    s.end_group();
    { s.begin_list() } -> std::same_as<bool>;
    s.end_list();
    { s.has_field(name) } -> std::same_as<bool>;  // query field existence
};

// ============================================================================
// Writable / Readable concepts (Layer 0 - assumed existing)
// These are marker concepts; actual verification happens at instantiation.
// ============================================================================

template <typename T, typename S>
concept WritableTo = Sink<S> && requires(S& sink, const T& t) {
    write(sink, t);
};

template <typename T, typename S>
concept ReadableFrom = Source<S> && requires(S& source, T& t) {
    { read(source, t) } -> std::same_as<bool>;
};

// ============================================================================
// Truthful concept (Layer 1)
// For types with runtime/state-machine data that should not be serialized.
// Only the "source of truth" is persisted via to_truth/from_truth.
// ============================================================================

template <typename T>
concept Truthful = requires(const T& ct, T& t) {
    { to_truth(ct) };  // returns some type
    requires std::default_initializable<std::remove_cvref_t<decltype(to_truth(ct))>>;
    from_truth(t, to_truth(ct));
};

// Generic write/read for Truthful types (makes them Writable/Readable)
template <Truthful T, Sink S>
void write(S& sink, const T& t) {
    write(sink, to_truth(t));
}

template <Truthful T, Source S>
auto read(S& source, T& t) -> bool {
    using Truth = std::remove_cvref_t<decltype(to_truth(t))>;
    Truth truth;
    if (!read(source, truth)) return false;
    from_truth(t, std::move(truth));
    return true;
}

// ============================================================================
// PatchKey concept
// ============================================================================

template <typename K>
concept PatchKey =
    std::equality_comparable<K> &&
    Hashable<K> &&
    requires(const K& k, std::string_view sv) {
        { to_string(k) } -> std::convertible_to<std::string>;
        { from_string(std::type_identity<K>{}, sv) } -> std::same_as<K>;
    };

// ============================================================================
// Protocol concepts (Layer 2 - checkpoint protocol)
// ============================================================================

template <typename T, typename SinkT, typename SourceT>
concept Emittable = Sink<SinkT> && Source<SourceT> &&
    requires(SinkT& sink, SourceT& source, const T& t, T& mut_t) {
        emit(sink, t);
        load(source, mut_t);
    };

template <typename T>
concept Scatterable = requires(const T& t) {
    typename T::patch_key_type;
    requires PatchKey<typename T::patch_key_type>;
    { patch_keys(t) };  // iterable of patch_key_type
    { patch_data(t, std::declval<typename T::patch_key_type>()) };  // returns Writable
};

template <typename T, typename SourceT>
concept Gatherable = Source<SourceT> && requires(T& t, typename T::patch_key_type key, SourceT& source, int rank, int num_ranks) {
    { patch_affinity(t, key, rank, num_ranks) } -> std::same_as<bool>;
    emplace_patch(t, key, source);  // reads patch data from source
};

template <typename T, typename SinkT, typename SourceT>
concept Checkpointable = Emittable<T, SinkT, SourceT> && Scatterable<T> && Gatherable<T, SourceT>;

// ============================================================================
// Checkpoint operations
// ============================================================================

namespace detail {

// Format traits
template <typename Format> struct format_traits;

template <>
struct format_traits<Binary> {
    using sink_type = binary_sink;
    using source_type = binary_source;
    static constexpr auto extension() { return ".bin"; }
    static constexpr auto open_mode() { return std::ios::binary; }
};

template <>
struct format_traits<Ascii> {
    using sink_type = ascii_sink;
    using source_type = ascii_source;
    static constexpr auto extension() { return ".dat"; }
    static constexpr auto open_mode() { return std::ios::openmode{}; }
};

inline std::filesystem::path patches_dir(const std::filesystem::path& dir) {
    return dir / "patches";
}

template <typename Format>
std::filesystem::path header_path(const std::filesystem::path& dir) {
    return dir / ("header" + std::string(format_traits<Format>::extension()));
}

template <typename Format, PatchKey K>
std::filesystem::path patch_path(const std::filesystem::path& dir, const K& key) {
    return patches_dir(dir) / (to_string(key) + std::string(format_traits<Format>::extension()));
}

} // namespace detail

// ----------------------------------------------------------------------------
// distributed_write: emit shared data + scatter patches
// ----------------------------------------------------------------------------

template <typename Format, Scatterable T>
void distributed_write(const std::filesystem::path& dir, const T& state, Format = {}) {
    namespace fs = std::filesystem;
    using traits = detail::format_traits<Format>;
    using sink_type = typename traits::sink_type;

    fs::create_directories(detail::patches_dir(dir));

    // 1. Emit shared/header data
    {
        std::ofstream file(detail::header_path<Format>(dir), traits::open_mode());
        if (!file) throw std::runtime_error("failed to open header file for writing");
        sink_type sink(file);
        emit(sink, state);
    }

    // 2. Scatter patches
    for (const auto& key : patch_keys(state)) {
        std::ofstream file(detail::patch_path<Format>(dir, key), traits::open_mode());
        if (!file) throw std::runtime_error("failed to open patch file for writing");
        sink_type sink(file);
        const auto& pdata = patch_data(state, key);
        write(sink, pdata);
    }
}

// ----------------------------------------------------------------------------
// distributed_read: load shared data + gather patches by affinity
// ----------------------------------------------------------------------------

template <typename Format, typename T>
    requires requires { typename T::patch_key_type; }
void distributed_read(const std::filesystem::path& dir, T& state, int rank, int num_ranks, Format = {}) {
    namespace fs = std::filesystem;
    using traits = detail::format_traits<Format>;
    using source_type = typename traits::source_type;
    auto ext = std::string(traits::extension());

    // 1. Load shared/header data (all ranks)
    {
        std::ifstream file(detail::header_path<Format>(dir), traits::open_mode());
        if (!file) throw std::runtime_error("failed to open header file for reading");
        source_type source(file);
        load(source, state);
    }

    // 2. Gather patches by affinity
    for (const auto& entry : fs::directory_iterator(detail::patches_dir(dir))) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ext) continue;

        auto stem = entry.path().stem().string();
        auto key = from_string(std::type_identity<typename T::patch_key_type>{}, stem);

        if (patch_affinity(state, key, rank, num_ranks)) {
            std::ifstream file(entry.path(), traits::open_mode());
            if (!file) throw std::runtime_error("failed to open patch file for reading");
            source_type source(file);
            emplace_patch(state, key, source);
        }
    }
}

// ----------------------------------------------------------------------------
// Overloads that accept format enum
// ----------------------------------------------------------------------------

template <Scatterable T>
void distributed_write(const std::filesystem::path& dir, const T& state, format fmt) {
    if (fmt == format::binary) {
        distributed_write(dir, state, Binary{});
    } else {
        distributed_write(dir, state, Ascii{});
    }
}

template <typename T>
    requires requires { typename T::patch_key_type; }
void distributed_read(const std::filesystem::path& dir, T& state, int rank, int num_ranks, format fmt) {
    if (fmt == format::binary) {
        distributed_read(dir, state, rank, num_ranks, Binary{});
    } else {
        distributed_read(dir, state, rank, num_ranks, Ascii{});
    }
}

} // namespace archive
