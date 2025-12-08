#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

namespace mist {
namespace parallel {

// =============================================================================
// Blocking queue (analogous to Rust channels)
// =============================================================================

template<typename T>
class blocking_queue {
public:
    void send(T item) {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _queue.push(std::move(item));
        }
        _cv.notify_one();
    }

    T recv() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] { return !_queue.empty(); });
        auto item = std::move(_queue.front());
        _queue.pop();
        return item;
    }

    std::optional<T> try_recv() {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_queue.empty()) {
            return std::nullopt;
        }
        auto item = std::move(_queue.front());
        _queue.pop();
        return item;
    }

private:
    std::queue<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cv;
};

} // namespace parallel
} // namespace mist
