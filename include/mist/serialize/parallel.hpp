#pragma once

// =============================================================================
// Parallel Serialization Concepts
// =============================================================================
//
// This header provides concepts for parallel/distributed serialization where
// data is written as: header + list of items (e.g., patches in domain decomposition).
//
// Writing:
//   - header(const T&) -> returns metadata to write once
//   - items(const T&)  -> returns iterable of items to write independently
//   - item_key(const Item&) -> returns unique key for the item (for filenames)
//
// Reading:
//   - header(T&) -> returns reference to metadata to populate
//   - items(T&)  -> returns container to populate with loaded items
//   - A predicate is passed at call site to filter which items to load
//
// Usage example:
//
//   struct patch_t { index_space_t space; std::vector<double> data; };
//   struct simulation_state_t {
//       config_t config;
//       std::vector<patch_t> patches;
//   };
//
//   // ADL functions for ParallelWrite/ParallelRead
//   auto header(const simulation_state_t& s) -> const config_t& { return s.config; }
//   auto items(const simulation_state_t& s) -> const auto& { return s.patches; }
//   auto header(simulation_state_t& s) -> config_t& { return s.config; }
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
concept ParallelWrite = requires(const T& t) {
    // header(t) returns something (metadata to serialize once)
    { header(t) };
    // items(t) returns an iterable of items
    { items(t) } -> std::ranges::range;
};

// =============================================================================
// ParallelRead concept
// =============================================================================

template<typename T>
concept ParallelRead = requires(T& t) {
    // header(t) returns reference to metadata to populate
    { header(t) };
    // items(t) returns container to populate
    { items(t) };
};

// =============================================================================
// Type traits for parallel serialization
// =============================================================================

template<ParallelWrite T>
using parallel_header_type = decltype(header(std::declval<const T&>()));

template<ParallelWrite T>
using parallel_items_type = decltype(items(std::declval<const T&>()));

template<ParallelWrite T>
using parallel_item_type = std::ranges::range_value_t<parallel_items_type<T>>;

// Key type for items (if HasItemKey)
template<HasItemKey T>
using item_key_type = decltype(item_key(std::declval<const T&>()));

} // namespace serialize
