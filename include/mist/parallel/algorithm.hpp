#pragma once

#include <concepts>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include "automaton.hpp"

namespace mist {
namespace parallel {

// =============================================================================
// Communicator for recording sends/recvs during exchange phase
// =============================================================================

template<typename Message>
class recording_communicator_t {
public:
    recording_communicator_t() = default;

    template<typename MessageData>
    void send(int dest, MessageData&& data) {
        _outgoing.emplace_back(dest, Message{_key, std::forward<MessageData>(data)});
    }

    void recv(int source) {
        _expected.push_back(source);
    }

    std::vector<std::pair<int, Message>> take_outgoing() {
        return std::move(_outgoing);
    }

    std::vector<int> take_expected() {
        return std::move(_expected);
    }

    void set_key(int key) {
        _key = key;
    }

private:
    int _key;
    std::vector<std::pair<int, Message>> _outgoing;
    std::vector<int> _expected;
};

// Forward declarations
template<typename Input, typename Message, typename Result, typename ExchangeFn, typename ComputeFn>
class two_stage_automaton_t;

template<typename Input, typename Result, typename TransformFn>
class transform_automaton_t;

// =============================================================================
// Two-stage algorithm type
// =============================================================================

template<typename Input, typename Message, typename Result, typename ExchangeFn, typename ComputeFn>
class two_stage_t {
public:
    using input_t = Input;
    using message_t = Message;
    using result_t = Result;

    two_stage_t(ExchangeFn exchange, ComputeFn compute)
        : _exchange(std::move(exchange))
        , _compute(std::move(compute))
    {}

    // Create an automaton for a specific key/input
    auto make_automaton(int key, Input input) const;

private:
    ExchangeFn _exchange;
    ComputeFn _compute;

    template<typename I, typename M, typename R, typename E, typename C>
    friend class two_stage_automaton_t;
};

// =============================================================================
// Automaton created by two_stage_t
// =============================================================================

template<typename Input, typename Message, typename Result, typename ExchangeFn, typename ComputeFn>
class two_stage_automaton_t {
public:
    using key_t = int;
    using message_t = Message;
    using value_t = Result;

    two_stage_automaton_t(int key, Input input, const ExchangeFn& exchange, const ComputeFn& compute)
        : _key(key)
        , _input(std::move(input))
        , _exchange(exchange)
        , _compute(compute)
    {
        // Execute exchange phase to record sends/recvs
        recording_communicator_t<Message> comm;
        comm.set_key(_key);
        _exchange(comm, _key, _input);
        _outgoing = comm.take_outgoing();
        _expected = comm.take_expected();
    }

    key_t key() const {
        return _key;
    }

    std::vector<std::pair<key_t, message_t>> messages() {
        return _outgoing;
    }

    status receive(message_t msg) {
        _received.push_back(std::move(msg));
        return _received.size() == _expected.size() ? status::eligible : status::ineligible;
    }

    value_t value() {
        return _compute(std::move(_input), std::move(_received));
    }

private:
    int _key;
    Input _input;
    ExchangeFn _exchange;
    ComputeFn _compute;
    std::vector<std::pair<int, Message>> _outgoing;
    std::vector<int> _expected;
    std::vector<Message> _received;
};

// =============================================================================
// Transform algorithm type (embarrassingly parallel)
// =============================================================================

template<typename Input, typename Result, typename TransformFn>
class transform_t {
public:
    using input_t = Input;
    using message_t = std::monostate;
    using result_t = Result;

    explicit transform_t(TransformFn fn)
        : _transform(std::move(fn))
    {}

    auto make_automaton(int key, Input input) const;

private:
    TransformFn _transform;
};

// =============================================================================
// Automaton created by transform_t
// =============================================================================

template<typename Input, typename Result, typename TransformFn>
class transform_automaton_t {
public:
    using key_t = int;
    using message_t = std::monostate;
    using value_t = Result;

    transform_automaton_t(int key, Input input, const TransformFn& fn)
        : _key(key)
        , _input(std::move(input))
        , _transform(fn)
    {}

    key_t key() const {
        return _key;
    }

    std::vector<std::pair<key_t, message_t>> messages() {
        return {};  // No messages
    }

    status receive(message_t) {
        return status::eligible;  // Always eligible
    }

    value_t value() {
        return _transform(std::move(_input));
    }

private:
    int _key;
    Input _input;
    TransformFn _transform;
};

// =============================================================================
// Implementation of make_automaton methods
// =============================================================================

template<typename Input, typename Message, typename Result, typename ExchangeFn, typename ComputeFn>
auto two_stage_t<Input, Message, Result, ExchangeFn, ComputeFn>::make_automaton(int key, Input input) const {
    return two_stage_automaton_t<Input, Message, Result, ExchangeFn, ComputeFn>{
        key,
        std::move(input),
        _exchange,
        _compute
    };
}

template<typename Input, typename Result, typename TransformFn>
auto transform_t<Input, Result, TransformFn>::make_automaton(int key, Input input) const {
    return transform_automaton_t<Input, Result, TransformFn>{
        key,
        std::move(input),
        _transform
    };
}

// =============================================================================
// Type extraction helpers for automatic deduction
// =============================================================================

// Extract types from lambda/function signatures for automatic type deduction.
// This allows two_stage(f, g) to deduce Input, Message, Result from the
// lambda parameter types instead of requiring explicit template parameters.

template<typename F>
struct function_signature;

template<typename C, typename R, typename... Args>
struct function_signature<R(C::*)(Args...) const> {
    using return_type = R;
    using arg_tuple = std::tuple<Args...>;
    template<std::size_t N>
    using arg = std::tuple_element_t<N, arg_tuple>;
};

template<typename C, typename R, typename... Args>
struct function_signature<R(C::*)(Args...)> {
    using return_type = R;
    using arg_tuple = std::tuple<Args...>;
    template<std::size_t N>
    using arg = std::tuple_element_t<N, arg_tuple>;
};

template<typename F>
    requires requires { &F::operator(); }
struct function_signature<F> : function_signature<decltype(&F::operator())> {};

template<typename T>
using clean_type = std::remove_cvref_t<T>;

// Extract Input from ExchangeFn signature: (Comm&, int, const Input&) -> void
template<typename ExchangeFn>
using extract_input_t = clean_type<typename function_signature<ExchangeFn>::template arg<2>>;

// Extract Result from ComputeFn signature: (Input, vector<Message>) -> Result
template<typename ComputeFn>
using extract_result_t = typename function_signature<ComputeFn>::return_type;

// Extract Message from ComputeFn signature: (Input, vector<Message>) -> Result
template<typename ComputeFn>
struct extract_message {
    using vec_type = clean_type<typename function_signature<ComputeFn>::template arg<1>>;
    using type = typename vec_type::value_type;
};

template<typename ComputeFn>
using extract_message_t = typename extract_message<ComputeFn>::type;

// =============================================================================
// Factory functions
// =============================================================================

// Create two-stage algorithm with explicit template parameters.
// Usage: two_stage<Input, Message, Result>(exchange_fn, compute_fn)
template<typename Input, typename Message, typename Result, typename ExchangeFn, typename ComputeFn>
auto two_stage(ExchangeFn&& exchange, ComputeFn&& compute) {
    return two_stage_t<Input, Message, Result, std::decay_t<ExchangeFn>, std::decay_t<ComputeFn>>{
        std::forward<ExchangeFn>(exchange),
        std::forward<ComputeFn>(compute)
    };
}

// Create two-stage algorithm with automatic type deduction.
// Usage: two_stage(exchange_fn, compute_fn)
//
// Requires explicit (non-auto) types in lambda parameters:
//   exchange_fn: [](recording_communicator_t<Message>& comm, int key, const Input& input)
//   compute_fn:  [](Input input, std::vector<Message> msgs) -> Result
template<typename ExchangeFn, typename ComputeFn>
    requires requires {
        typename extract_input_t<std::decay_t<ExchangeFn>>;
        typename extract_message_t<std::decay_t<ComputeFn>>;
        typename extract_result_t<std::decay_t<ComputeFn>>;
    }
auto two_stage(ExchangeFn&& exchange, ComputeFn&& compute) {
    using Input = extract_input_t<std::decay_t<ExchangeFn>>;
    using Message = extract_message_t<std::decay_t<ComputeFn>>;
    using Result = extract_result_t<std::decay_t<ComputeFn>>;

    return two_stage_t<Input, Message, Result, std::decay_t<ExchangeFn>, std::decay_t<ComputeFn>>{
        std::forward<ExchangeFn>(exchange),
        std::forward<ComputeFn>(compute)
    };
}

// Create embarrassingly parallel transform with explicit template parameters.
// Usage: transform<Input, Result>(transform_fn)
template<typename Input, typename Result, typename TransformFn>
auto transform(TransformFn&& fn) {
    return transform_t<Input, Result, std::decay_t<TransformFn>>{
        std::forward<TransformFn>(fn)
    };
}

// =============================================================================
// Execute algorithm over a vector of inputs
// =============================================================================

template<typename Algorithm, Scheduler S>
auto execute(
    const Algorithm& algo,
    S& scheduler,
    std::vector<typename Algorithm::input_t> inputs
) -> std::vector<typename Algorithm::result_t> {
    using message_t = typename Algorithm::message_t;

    // Create automatons from algorithm + inputs
    std::vector<decltype(algo.make_automaton(0, std::move(inputs[0])))> automatons;
    automatons.reserve(inputs.size());

    for (int key = 0; key < static_cast<int>(inputs.size()); ++key) {
        automatons.push_back(algo.make_automaton(key, std::move(inputs[key])));
    }

    // Execute using existing executor
    local_communicator<message_t> comm(static_cast<int>(inputs.size()));
    return mist::parallel::execute(std::move(automatons), comm, scheduler);
}

} // namespace parallel
} // namespace mist
