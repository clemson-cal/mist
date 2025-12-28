#pragma once

// =============================================================================
// Parallel Serialization Concepts
// =============================================================================
//
// This header provides concepts for parallel/distributed serialization where
// data is written as: header + list of items (e.g., patches in domain decomposition).
//
// Writing:
//   - serialize_header(writer, const T&) -> serializes metadata once
//   - items(const T&) -> returns iterable of items to write independently
//   - item_key(const Item&) -> returns unique key for the item (for filenames)
//
// Reading:
//   - deserialize_header(reader, T&) -> deserializes metadata
//   - items(T&) -> returns container to populate with loaded items
//   - A predicate is passed at call site to filter which items to load
//
// Usage example:
//
//   struct patch_t { index_space_t space; std::vector<double> data; };
//   struct simulation_state_t {
//       double time;
//       std::vector<patch_t> patches;
//   };
//
//   // Explicit header serialization
//   template<ArchiveWriter A>
//   void serialize_header(A& ar, const simulation_state_t& s) {
//       serialize(ar, "time", s.time);
//   }
//
//   template<ArchiveReader A>
//   auto deserialize_header(A& ar, simulation_state_t& s) -> bool {
//       return deserialize(ar, "time", s.time);
//   }
//
//   // Items access
//   auto items(const simulation_state_t& s) -> const auto& { return s.patches; }
//   auto items(simulation_state_t& s) -> auto& { return s.patches; }
//
//   // Item key for filename generation
//   auto item_key(const patch_t& p) -> std::string { return to_string(p.space); }
//
//   // Reading with predicate that captures execution context
//   parallel_read(path, state, [&ctx](const auto& key) {
//       return ctx.owns(key);
//   });
//
// =============================================================================

#include <concepts>
#include <ranges>
#include <type_traits>

#include "binary_writer.hpp"
#include "binary_reader.hpp"

namespace serialize {

// =============================================================================
// Item key concept - items need a unique key for filename generation
// =============================================================================

template<typename T>
concept HasItemKey = requires(const T& item) {
    { item_key(item) } -> std::convertible_to<std::string>;
};

// =============================================================================
// ParallelWrite concept
// =============================================================================

template<typename T>
concept ParallelWrite = requires(const T& t, binary_writer& w) {
    // serialize_header writes metadata
    { serialize_header(w, t) };
    // items(t) returns an iterable of items
    { items(t) } -> std::ranges::range;
};

// =============================================================================
// ParallelRead concept
// =============================================================================

template<typename T>
concept ParallelRead = requires(T& t, binary_reader& r) {
    // deserialize_header reads metadata
    { deserialize_header(r, t) } -> std::same_as<bool>;
    // items(t) returns container to populate
    { items(t) };
};

// =============================================================================
// Type traits for parallel serialization
// =============================================================================

template<ParallelWrite T>
using parallel_items_type = decltype(items(std::declval<const T&>()));

template<ParallelWrite T>
using parallel_item_type = std::ranges::range_value_t<parallel_items_type<T>>;

// Key type for items (if HasItemKey)
template<HasItemKey T>
using item_key_type = decltype(item_key(std::declval<const T&>()));

} // namespace serialize
