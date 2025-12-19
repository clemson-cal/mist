#pragma once

#include <cstddef>
#include <span>
#include <vector>
#include "command.hpp"

namespace mist::driver {

// =============================================================================
// communicator_t - abstract interface for inter-process communication
// =============================================================================

struct communicator_t {
    virtual ~communicator_t() = default;

    // Rank and size
    virtual auto rank() const -> int = 0;
    virtual auto size() const -> int = 0;
    auto is_root() const -> bool { return rank() == 0; }

    // Point-to-point (for halo exchange)
    virtual void sendrecv(int peer,
                          std::span<const std::byte> send,
                          std::span<std::byte> recv) = 0;

    // Collective reductions
    virtual auto allreduce_min(double local) -> double = 0;
    virtual auto allreduce_max(double local) -> double = 0;
    virtual auto allreduce_sum(double local) -> double = 0;

    // Command broadcast (root sends, others receive)
    virtual void broadcast_command(command_t& cmd) = 0;

    // Synchronization
    virtual void barrier() = 0;
};

// =============================================================================
// null_communicator_t - single-process implementation (no-op)
// =============================================================================

struct null_communicator_t : communicator_t {
    auto rank() const -> int override { return 0; }
    auto size() const -> int override { return 1; }

    void sendrecv(int /*peer*/,
                  std::span<const std::byte> /*send*/,
                  std::span<std::byte> /*recv*/) override {
        // No peers in single-process mode
    }

    auto allreduce_min(double local) -> double override { return local; }
    auto allreduce_max(double local) -> double override { return local; }
    auto allreduce_sum(double local) -> double override { return local; }

    void broadcast_command(command_t& /*cmd*/) override {
        // Command already present, nothing to broadcast
    }

    void barrier() override {
        // Nothing to synchronize
    }
};

// =============================================================================
// Factory function
// =============================================================================

inline auto make_null_communicator() -> std::unique_ptr<communicator_t> {
    return std::make_unique<null_communicator_t>();
}

} // namespace mist::driver
