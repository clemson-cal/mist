#pragma once

#include <concepts>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mist {
namespace parallel {

// =============================================================================
// Scheduler concept
// =============================================================================

template<typename S>
concept Scheduler = requires(S& scheduler) {
    { scheduler.spawn([](){}) } -> std::same_as<void>;
};

// =============================================================================
// Sequential scheduler (executes tasks immediately)
// =============================================================================

class sequential_scheduler_t {
public:
    template<typename F>
    void spawn(F&& task) {
        task();
    }
};

// =============================================================================
// Thread pool scheduler
// =============================================================================

class thread_pool_t {
public:
    explicit thread_pool_t(std::size_t num_threads) {
        _workers.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            _workers.emplace_back([this] { worker_loop(); });
        }
    }

    ~thread_pool_t() {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _stop = true;
        }
        _cv.notify_all();
        for (auto& worker : _workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<typename F>
    void spawn(F&& task) {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            auto shared_task = std::make_shared<std::decay_t<F>>(std::forward<F>(task));
            _tasks.push([shared_task]() mutable { (*shared_task)(); });
        }
        _cv.notify_one();
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(_mutex);
                _cv.wait(lock, [this] {
                    return _stop || !_tasks.empty();
                });
                if (_stop && _tasks.empty()) {
                    return;
                }
                if (!_tasks.empty()) {
                    task = std::move(_tasks.front());
                    _tasks.pop();
                }
            }
            if (task) {
                task();
            }
        }
    }

    std::vector<std::thread> _workers;
    std::queue<std::function<void()>> _tasks;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _stop = false;
};

// Backward compatibility alias
using threadpool = thread_pool_t;

} // namespace parallel
} // namespace mist
