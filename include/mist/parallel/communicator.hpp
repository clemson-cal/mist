#pragma once

#include <mutex>
#include <optional>
#include <vector>

namespace mist {
namespace parallel {

// =============================================================================
// Local communicator (in-process message passing)
// =============================================================================

// The local_communicator simulates distributed message passing within a single
// process. Each "rank" has a mailbox, and automata can send messages to other
// ranks. This is useful for:
// 1. Testing distributed algorithms without MPI
// 2. Domain decomposition within a single process
// 3. Simulating message-passing patterns

template<typename MessageT>
class local_communicator {
public:
    explicit local_communicator(int size)
        : _size(size)
        , _mailboxes(size)
    {}

    int size() const {
        return _size;
    }

    void send(int dest, MessageT message) {
        std::unique_lock<std::mutex> lock(_mutex);
        _mailboxes[dest].push_back(std::move(message));
    }

    std::optional<MessageT> try_recv(int rank) {
        std::unique_lock<std::mutex> lock(_mutex);
        auto& mailbox = _mailboxes[rank];
        if (mailbox.empty()) {
            return std::nullopt;
        }
        auto msg = std::move(mailbox.front());
        mailbox.erase(mailbox.begin());
        return msg;
    }

    bool has_messages(int rank) const {
        std::unique_lock<std::mutex> lock(_mutex);
        return !_mailboxes[rank].empty();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(_mutex);
        for (auto& mailbox : _mailboxes) {
            mailbox.clear();
        }
    }

private:
    int _size;
    std::vector<std::vector<MessageT>> _mailboxes;
    mutable std::mutex _mutex;
};

} // namespace parallel
} // namespace mist
