#pragma once

#include <concepts>
#include <condition_variable>
#include <functional>
#include <memory>
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

// =============================================================================
// Configurable scheduler (switches between sequential and thread pool)
// =============================================================================

class scheduler_t {
public:
    scheduler_t() = default;

    void set_num_threads(std::size_t n) {
        _pool.reset();
        _num_threads = n;
        if (n > 0) {
            _pool = std::make_unique<thread_pool_t>(n);
        }
    }

    std::size_t num_threads() const {
        return _num_threads;
    }

    template<typename F>
    void spawn(F&& task) {
        if (_pool) {
            _pool->spawn(std::forward<F>(task));
        } else {
            task();
        }
    }

private:
    std::size_t _num_threads = 0;
    std::unique_ptr<thread_pool_t> _pool;
};

} // namespace parallel
} // namespace mist
