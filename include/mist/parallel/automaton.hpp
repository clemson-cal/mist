#pragma once

#include <concepts>
#include <utility>
#include <vector>

namespace mist {
namespace parallel {

// =============================================================================
// Status enum for task eligibility
// =============================================================================

enum class status {
    eligible,
    ineligible,
};

// =============================================================================
// Automaton concept
// =============================================================================

template<typename A>
concept Automaton = requires(
    A a,
    const A ca,
    typename A::message_t msg
) {
    typename A::key_t;
    typename A::message_t;
    typename A::value_t;

    { ca.key() } -> std::same_as<typename A::key_t>;
    { a.messages() } -> std::same_as<std::vector<std::pair<typename A::key_t, typename A::message_t>>>;
    { a.receive(msg) } -> std::same_as<status>;
    { std::move(a).value() } -> std::same_as<typename A::value_t>;
};

} // namespace parallel
} // namespace mist
