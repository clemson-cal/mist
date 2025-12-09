#pragma once

#include <thread>
#include <unordered_map>
#include <vector>
#include "automaton.hpp"
#include "communicator.hpp"
#include "queue.hpp"
#include "scheduler.hpp"

namespace mist {
namespace parallel {

// =============================================================================
// Coordinate function (analogous to Rust's coordinate)
// =============================================================================

// Core coordination logic: delivers messages and spawns eligible tasks.
// The sink function is called for each eligible automaton.

template<Automaton A, typename Sink>
void coordinate(
    std::vector<A> automata,
    local_communicator<typename A::message_t>& comm,
    Sink&& sink
) {
    using key_t = typename A::key_t;
    using message_t = typename A::message_t;

    std::unordered_map<key_t, A> seen;
    std::unordered_map<key_t, std::vector<message_t>> undelivered;

    for (auto& a : automata) {
        // For each of A's messages, either deliver it to the recipient peer
        // if the peer has already been seen, or put it in the undelivered box
        for (auto& [dest, data] : a.messages()) {
            auto rank = static_cast<int>(dest);
            comm.send(rank, std::move(data));

            auto it = seen.find(dest);
            if (it != seen.end()) {
                // Deliver immediately if recipient already seen
                while (auto msg = comm.try_recv(rank)) {
                    if (it->second.receive(std::move(*msg)) == status::eligible) {
                        auto task = std::move(it->second);
                        seen.erase(it);
                        sink(std::move(task));
                        break;
                    }
                }
            }
        }

        // Deliver any messages addressed to A that arrived previously
        auto key = a.key();
        auto rank = static_cast<int>(key);
        bool eligible = false;

        while (auto msg = comm.try_recv(rank)) {
            if (a.receive(std::move(*msg)) == status::eligible) {
                eligible = true;
            }
        }

        if (eligible) {
            sink(std::move(a));
        } else {
            seen.emplace(key, std::move(a));
        }
    }

    // Process remaining pending automata as messages arrive
    while (!seen.empty()) {
        bool made_progress = false;

        for (auto it = seen.begin(); it != seen.end();) {
            auto rank = static_cast<int>(it->first);
            bool eligible = false;

            while (auto msg = comm.try_recv(rank)) {
                if (it->second.receive(std::move(*msg)) == status::eligible) {
                    eligible = true;
                }
            }

            if (eligible) {
                auto task = std::move(it->second);
                it = seen.erase(it);
                sink(std::move(task));
                made_progress = true;
            } else {
                ++it;
            }
        }

        if (!made_progress && !seen.empty()) {
            std::this_thread::yield();
        }
    }
}

// =============================================================================
// Execute with local communicator and scheduler
// =============================================================================

template<Automaton A, Scheduler S>
std::vector<typename A::value_t> execute(
    std::vector<A> automata,
    local_communicator<typename A::message_t>& comm,
    S& scheduler
) {
    using key_t = typename A::key_t;
    using value_t = typename A::value_t;

    blocking_queue<std::pair<key_t, value_t>> results_queue;
    std::size_t total = automata.size();

    // Create sink that spawns tasks and sends results to queue
    auto sink = [&](A a) {
        scheduler.spawn([&results_queue, a = std::move(a)]() mutable {
            auto key = a.key();
            results_queue.send({key, std::move(a).value()});
        });
    };

    // Coordinate message delivery and task spawning
    coordinate(std::move(automata), comm, sink);

    // Collect all results from queue (blocks until all arrive)
    std::vector<value_t> results(total);
    for (std::size_t i = 0; i < total; ++i) {
        auto [key, value] = results_queue.recv();
        results[static_cast<std::size_t>(key)] = std::move(value);
    }

    return results;
}

} // namespace parallel
} // namespace mist
