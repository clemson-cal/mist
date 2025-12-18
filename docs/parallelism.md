# Parallelism

The parallelism library (`mist/parallel.hpp`) provides task scheduling, pipeline execution, and inter-task communication for domain-decomposed physics simulations.

## Scheduler

Task scheduling with runtime-switchable parallelism.

### Types

- `sequential_scheduler_t` — executes tasks immediately in calling thread
- `thread_pool_t` — worker threads with task queue
- `scheduler_t` — wrapper that switches between sequential and parallel

### Usage

```cpp
parallel::scheduler_t sched;
sched.set_num_threads(4);  // 0 = sequential, >0 = parallel

sched.spawn([&] {
    // task runs in thread pool
});
```

The same code works whether `num_threads` is 0 (sequential) or >0 (parallel), enabling easy debugging.

## Pipeline

Pipelines coordinate multiple peers (patches) through a sequence of stages with automatic synchronization.

### Stage Types

**ComputeStage** — per-peer parallel computation:
```cpp
struct my_compute_t {
    static constexpr const char* name = "my_compute";

    auto value(patch_t p) const -> patch_t {
        // local computation
        return p;
    }
};
```

**ExchangeStage** — ghost zone communication:
```cpp
struct ghost_exchange_t {
    static constexpr const char* name = "ghost_exchange";
    using space_t = index_space_t<1>;
    using buffer_t = array_view_t<double, 1>;

    auto provides(const patch_t& p) const -> space_t {
        return p.interior;
    }

    void need(patch_t& p, auto request) const {
        request(p.cons[left_guard]);
        request(p.cons[right_guard]);
    }

    auto data(const patch_t& p) const -> array_view_t<const double, 1> {
        return p.cons[p.interior];
    }
};
```

**ReduceStage** — global reduction with broadcast:
```cpp
struct global_dt_t {
    static constexpr const char* name = "global_dt";
    using value_type = double;

    static double init() {
        return std::numeric_limits<double>::max();
    }

    double reduce(double acc, const patch_t& p) const {
        return std::min(acc, p.dt);
    }

    void finalize(double dt, patch_t& p) const {
        p.dt = dt;
    }
};
```

### Pipeline Construction

```cpp
auto pipeline = parallel::pipeline(
    ghost_exchange_t{},
    compute_local_dt_t{cfl, v, dx},
    global_dt_t{},
    flux_and_update_t{}
);

parallel::execute(pipeline, patches, scheduler, profiler);
```

### Stage Composition

Fuse multiple compute stages for better cache utilization:
```cpp
auto fused = parallel::compose(stage_a, stage_b, stage_c);
```

### Synchronization

Stages have implicit barriers:
- **Exchange**: All peers wait for guard data before proceeding
- **Reduce**: All peers contribute before broadcast
- **Compute**: Peers proceed independently (barrier only at stage boundaries)

## Queue

Thread-safe message passing.

```cpp
parallel::blocking_queue<T> queue;

// Producer thread
queue.send(value);

// Consumer thread
auto value = queue.recv();           // blocking
auto maybe = queue.try_recv();       // non-blocking, returns optional
```

Used internally by pipeline execution to collect parallel compute results.

## Profiler

Timing measurement for pipeline stages.

### Types

- `null_profiler_t` — no-op (zero overhead when disabled)
- `profiler_t` — accumulates timing data

### Usage

```cpp
perf::profiler_t profiler;

profiler.start();
// ... work ...
profiler.record("stage_name");

// Get results
auto data = profiler.data();  // map<string, profile_entry_t>
for (auto& [name, entry] : data) {
    std::cout << name << ": " << entry.time << "s (" << entry.count << " calls)\n";
}
```

### Integration with Pipeline

Pipeline execution automatically records timing for each stage:
```cpp
parallel::execute(pipeline, patches, scheduler, profiler);
// profiler now contains timing for each named stage
```

---

## Execution Algorithm

The pipeline execution algorithm coordinates peers through stages with eager advancement—peers proceed as soon as their dependencies are satisfied.

### Per-Stage Behavior

**Exchange stage:**
1. Collect guard requests from each peer
2. Match requests to providers
3. Copy data when provider and requester are both ready
4. Peer advances when: all guards filled AND all requesters served

**Compute stage:**
1. Spawn compute task via scheduler
2. Peer advances when: task complete

**Reduce stage:**
1. Sequential fold across all peers
2. Broadcast result to all peers

### Why Wait for Requesters

A peer cannot proceed until all potential requesters are served because:
1. Next stage might modify the peer's data
2. Late requesters would receive corrupt data

### Complexity

- **Space**: O(stages × peers × max_requests)
- **Time per iteration**: O(stages × peers² × max_requests) worst case

## Example: 1D Advection Pipeline

```cpp
// Define pipeline stages
struct ghost_exchange_t { ... };
struct compute_dt_t { ... };
struct global_dt_t { ... };
struct flux_update_t { ... };

// Build and execute pipeline
auto pipeline = parallel::pipeline(
    ghost_exchange_t{},      // exchange ghost zones
    compute_dt_t{cfl, dx},   // compute local timestep
    global_dt_t{},           // reduce to global timestep
    flux_update_t{}          // compute fluxes and update
);

// Execute one timestep
parallel::execute(pipeline, patches, scheduler, profiler);

// Get timing data
for (auto& [name, entry] : profiler.data()) {
    std::cout << name << ": " << entry.time << "s\n";
}
```
