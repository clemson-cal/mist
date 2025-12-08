# Algorithm API: Final Design

## Overview

The **Algorithm API** separates reusable algorithms from data, providing a clean factory-based interface for parallel computations.

## API Design

### Two-Stage Algorithm (Message-Passing)

```cpp
auto algo = two_stage<input_t, message_t, result_t>(
    // Exchange: send/recv declarations
    [](auto& comm, int key, const input_t& input) {
        if (key > 0) {
            comm.send(key - 1, make_message(input));
            comm.recv(key - 1);
        }
    },
    // Compute: process with received messages
    [](input_t input, std::vector<message_t> msgs) -> result_t {
        return compute(input, msgs);
    }
);

std::vector<result_t> results = execute(algo, scheduler, state_vector);
```

### Transform Algorithm (Embarrassingly Parallel)

```cpp
auto algo = transform<input_t, result_t>(
    [](input_t input) -> result_t {
        return process(input);
    }
);

std::vector<result_t> results = execute(algo, scheduler, state_vector);
```

## Key Features

✅ **Separation of Algorithm and Data**
- Define algorithm once
- Execute on different data
- Algorithm is reusable

✅ **Simple Factory Functions**
- `two_stage(exchange_fn, compute_fn)` for message-passing
- `transform(fn)` for embarrassingly parallel
- No chaining, just function calls

✅ **Type Parameters Explicit**
```cpp
two_stage<input_t, message_t, result_t>(...)
transform<input_t, result_t>(...)
```

✅ **Stateless Communicator**
- `comm.send(dest, data)` - record send
- `comm.recv(source)` - expect receive
- `key` parameter identifies rank

✅ **Execute Takes Vector**
```cpp
std::vector<input_t> inputs = {...};
std::vector<result_t> outputs = execute(algo, scheduler, inputs);
```

## Example: Advection

```cpp
auto make_advection_algorithm(const global_config_t& cfg, int nranks) {
    return two_stage<subdomain_state_t, ghost_message_t, subdomain_state_t>(
        // Exchange phase
        [&cfg, nranks](auto& comm, int key, const subdomain_state_t& s) {
            if (key > 0) {
                comm.send(key - 1, make_left_boundary(s, cfg));
                comm.recv(key - 1);
            }
            if (key < nranks - 1) {
                comm.send(key + 1, make_right_boundary(s, cfg));
                comm.recv(key + 1);
            }
        },
        // Compute phase
        [&cfg](subdomain_state_t s, std::vector<ghost_message_t> msgs) {
            for (auto& msg : msgs) {
                insert_ghost_data(s, msg);
            }
            return timestep(s, cfg);
        }
    );
}

// Use it
auto algo = make_advection_algorithm(cfg, nranks);
auto states = decompose_domain(cfg, nranks);

for (int step = 0; step < nsteps; ++step) {
    states = execute(algo, pool, std::move(states));
}
```

## Performance

| Implementation | Avg Mzps | API Style |
|----------------|----------|-----------|
| Manual Automaton | 7,267 | State machine |
| Builder Pattern | 7,694 | `.send_to().await_from().then()` |
| Staged Pattern | 7,379 | `.exchange().then()` |
| **Algorithm API** | **7,443** | **`two_stage(exchange, compute)`** |
| Coroutines | 3,395 | `co_await` |

All Automaton-based approaches achieve ~7-8 Gzps!

## Design Rationale

### Why Separate Algorithm from Data?

**Reusability**:
```cpp
auto algo = make_advection_algorithm(cfg, nranks);

// Use on different initial conditions
auto ic1 = sine_wave(cfg, nranks);
auto ic2 = gaussian(cfg, nranks);

auto result1 = execute(algo, pool, ic1);
auto result2 = execute(algo, pool, ic2);
```

**Testability**:
```cpp
// Test algorithm with small data
std::vector<small_input> test_data = {...};
auto test_results = execute(algo, sequential_scheduler, test_data);
assert_correct(test_results);

// Use same algorithm on production data
auto results = execute(algo, thread_pool, production_data);
```

**Composition**:
```cpp
// Algorithms are values - can store, pass around
std::vector<Algorithm> pipeline = {
    two_stage(...),  // Step 1
    transform(...),  // Step 2
    two_stage(...)   // Step 3
};

auto data = initial_state;
for (auto& algo : pipeline) {
    data = execute(algo, pool, std::move(data));
}
```

### Why Factory Functions?

**No Chaining Complexity**:
```cpp
// Simple: just call a function
auto algo = two_stage(exchange, compute);

// vs builder pattern:
auto algo = builder().exchange(...).compute(...);
```

**Type Parameters Explicit**:
```cpp
two_stage<input_t, message_t, result_t>(...)
//       ^^^^^^^  ^^^^^^^^^^  ^^^^^^^^
//       Clear contract
```

**Easy to Extend**:
```cpp
// Future: three_stage, map_reduce, etc.
auto algo = three_stage(exchange1, compute1, exchange2, compute2);
auto algo = map_reduce(map_fn, reduce_fn);
```

### Why `key` Parameter?

**Simplicity**:
```cpp
[](auto& comm, int key, const input_t& s) {
    // key is just another parameter
    if (key > 0) comm.send(key - 1, ...);
}
```

**vs Communicator Method**:
```cpp
[](auto& comm, const input_t& s) {
    int key = comm.rank();  // Extra indirection
}
```

**Testability**:
```cpp
// Easy to test exchange function
exchange_fn(mock_comm, 5, test_input);  // Test as rank 5
```

## Comparison with Previous Approaches

### vs Builder Pattern

**Builder**:
```cpp
make_automaton(key, state)
    .send_to_if(key > 0, key-1, make_msg)
    .await_from_if(key > 0, key-1, handle)
    .then(compute)
```

**Algorithm API**:
```cpp
two_stage<...>(
    [](auto& comm, int key, const auto& s) {
        if (key > 0) {
            comm.send(key-1, make_msg(s));
            comm.recv(key-1);
        }
    },
    compute
)
```

Advantages:
- ✅ Procedural send/recv (more familiar)
- ✅ Algorithm separate from data
- ✅ No conditional methods (`_if`)

### vs Staged Pattern

**Staged**:
```cpp
staged(key, state)
    .exchange([](auto& comm, const auto& s) { ... })
    .then([](auto s, auto msgs) { ... })
```

**Algorithm API**:
```cpp
two_stage<...>(
    [](auto& comm, int key, const auto& s) { ... },
    [](auto s, auto msgs) { ... }
)
```

Advantages:
- ✅ No chaining - just function call
- ✅ Reusable (not tied to specific state)
- ✅ Key is explicit parameter

### vs Coroutines

**Coroutines**:
```cpp
task<state_t> step(state_t s, comm& c) {
    auto msg = co_await c.recv(source);
    return compute(s, msg);
}
```

**Algorithm API**:
```cpp
two_stage<...>(
    [](auto& comm, int key, const auto& s) {
        comm.recv(source);
    },
    [](auto s, auto msgs) {
        return compute(s, msgs[0]);
    }
)
```

Advantages:
- ✅ 2.2x faster (no coroutine overhead)
- ✅ No heap allocation
- ✅ Works in C++17

## Future Extensions

### Automatic Type Deduction

Currently:
```cpp
two_stage<input_t, message_t, result_t>(exchange, compute)
```

Future (with C++20 concepts):
```cpp
two_stage(exchange, compute)  // Deduces all types from lambdas
```

### Multi-Stage

```cpp
multi_stage<input_t, msg1_t, msg2_t, result_t>(
    exchange1, compute1,
    exchange2, compute2
)
```

### Collective Operations

```cpp
all_reduce<input_t, result_t>(
    local_fn,    // Compute local contribution
    reduce_op    // Combine (e.g., std::plus, std::max)
)
```

### Async Execution

```cpp
std::future<std::vector<result_t>> future = execute_async(algo, pool, inputs);
```

## Recommendation

**Use the Algorithm API for new code!**

It provides the best combination of:
- Performance (7.4 Gzps)
- Simplicity (factory functions, no chaining)
- Reusability (algorithm separate from data)
- Familiarity (procedural send/recv)

Perfect for message-passing parallel algorithms in scientific computing.
