# Mist

A lightweight C++20 header-only library for deploy-anywhere array transformations and physics simulations.

Mist is an evolution of [Vapor](https://github.com/clemson-cal/vapor), a library with similar goals for HPC physics applications.

## Features

- **Core library** (`mist/core.hpp`): Multi-dimensional arrays, index spaces, transforms, and parallel execution
- **Driver library** (`mist/driver.hpp`): Time-stepping and scheduled output management for physics simulations
- **Header-only**: No compilation required, just include and go
- **CUDA compatible**: All functions work on both CPU and GPU (CUDA 12+)
- **Zero dependencies**: Pure C++20 standard library

## Quick Start

```bash
# Copy headers to your project
cp -r include/mist /path/to/your/project/

# Or clone and use directly
git clone https://github.com/yourusername/mist.git
cd mist/examples/advection-1d
make
./advection-1d
```

## CUDA Compatibility

Mist is fully compatible with CUDA 12+ and can be used in both host and device code. All functions are annotated with `__host__ __device__` when compiled with nvcc.

## Data structures
All data structures use overloaded free functions rather than member functions. Data members are public but start with an underscore.

- `vec_t<T, S>` is a statically sized array type with element type `T` and size `S`
  * aliases `dvec_t`, `ivec_t` and, `uvec_t` for types `double`, `int`, and `unsigned int`
  * constructor e.g. `vec(0.4, 1.0, 2.0) -> vec_t<double, 3>` detects `T` using `std::common_type`
  * constructors
    + `dvec`
    + `ivec`
    + `uvec`
    + `range<S> -> uvec_t<S>`
  * operators
    + `vec_t<T, S> + vec_t<U, S>`
    + `vec_t<T, S> - vec_t<U, S>`
    + `vec_t<T, S> * U` and `U * vec_t<T, S>`
    + `vec_t<T, S> / U`
    + `==`
  * free functions
    + `dot(vec<T, S>, vec<U, S>)` - dot product
    + `map(vec_t<T, S>, [] (T) -> U { ... })` - element-wise transformation
    + `sum(vec_t<T, S>)` - sum of all elements, returns `T`
    + `product(vec_t<T, S>)` - product of all elements, returns `T`
    + `any(vec_t<bool, S>)` - true if any element is true, returns `bool`
    + `all(vec_t<bool, S>)` - true if all elements are true, returns `bool`
- `index_space_t<S>` is `start: ivec_t<S>` and a `shape: uvec_t<S>`
  * constructors
    + `index_space(start, shape)` - creates an index_space_t with given start and shape
    + Example: `auto space = index_space(ivec(0, 0), uvec(10, 20));`
  * operators
    + `==`
  * free functions
    + `start(space)` - returns `_start`
    + `shape(space)` - returns `_shape`
    + `size(space)` - returns total number of elements
    + `contains(space, index)` - returns `bool`, checks if index is within bounds
  * iteration
    + `begin(space)` and `end(space)` - iterate over all indices in the space
    + Example: `for (auto index : space) { ... }` iterates through all valid indices
    + `for_each(space, func)` - execute `func(index)` for every index in the space
    + `for_each(space, func, exec_mode)` where `exec_mode` is one of `exec::cpu`, `exec::omp`, `exec::gpu`
  * reductions
    + `map_reduce(space, init, map, reduce_op)` - map-reduce over all indices
    + `map_reduce(space, init, map, reduce_op, exec_mode)` - with specified execution policy
    + `map` signature: `(ivec_t<S> index) -> T` - transforms index to value
    + `reduce_op` signature: `(T, T) -> T` - binary associative reduction operator
    + Example: `auto sum = map_reduce(space, 0, [&buf](auto idx) { return ndread(buf, space, idx); }, std::plus<>{});`
    + Example: `auto max = map_reduce(space, -INF, [&buf](auto idx) { return ndread(buf, space, idx); }, [](auto a, auto b) { return std::max(a, b); });`

## Multi-dimensional indexing functions
All indices are **absolute** (relative to the origin `[0, 0, ...]`). For example, with `_start = [5, 10]` and `_shape = [10, 20]`, valid indices range from `[5, 10]` to `[14, 29]` (i.e., `_start` to `_start + _shape - 1`).

- `ndoffset(space, index)` - Compute flat buffer offset from multi-dimensional index
  * `space: index_space_t<S>` - The index space defining the array dimensions
  * `index: ivec_t<S>` - Absolute multi-dimensional index
  * Returns `std::size_t` - Flat offset into buffer (row-major ordering)
  * Example: `ndoffset(space, ivec(7, 13))` with `start = [5, 10]` and `shape = [10, 20]` returns `(7-5) * 20 + (13-10) = 43`

- `ndindex(space, offset)` - Convert flat buffer offset to multi-dimensional index
  * `space: index_space_t<S>` - The index space defining the array dimensions
  * `offset: std::size_t` - Flat offset into buffer
  * Returns `ivec_t<S>` - Absolute multi-dimensional index
  * Inverse of `ndoffset`: `ndindex(space, ndoffset(space, index)) == index`

- `ndread(data, space, index)` - Read element from buffer using multi-dimensional index
  * `data: const T*` - Pointer to flat buffer
  * `space: index_space_t<S>` - Index space defining dimensions
  * `index: ivec_t<S>` - Multi-dimensional index
  * Returns `T` - Value at that index
  * Example: `ndread(buffer, space, ivec(1, 2))` reads element at row 1, column 2

- `ndwrite(data, space, index, value)` - Write element to buffer using multi-dimensional index
  * `data: T*` - Pointer to flat buffer
  * `space: index_space_t<S>` - Index space defining dimensions
  * `index: ivec_t<S>` - Multi-dimensional index
  * `value: T` - Value to write
  * Example: `ndwrite(buffer, space, ivec(1, 2), 42.0)` writes to row 1, column 2

- `ndread_soa<T, N>(data, space, index)` - Read a `vec_t<T, N>` from SoA buffer
  * Template parameters: `T` (element type), `N` (vector size)
  * `data: const T*` - Pointer to flat buffer with all components stored contiguously
  * `space: index_space_t<S>` - Index space defining spatial dimensions
  * `index: ivec_t<S>` - Multi-dimensional spatial index
  * Returns `vec_t<T, N>` - Vector with components gathered from memory
  * Layout: Component `i` of vector at `index` is at offset `i * size(space) + ndoffset(space, index)`
  * Component-major ordering: For a 2D grid `[10, 20]` storing 3-vectors (200 total positions):
    - All x-components: `[x₀, x₁, x₂, ..., x₁₉₉]`
    - All y-components: `[y₀, y₁, y₂, ..., y₁₉₉]`
    - All z-components: `[z₀, z₁, z₂, ..., z₁₉₉]`
  * Usage: `auto v = ndread_soa<double, 3>(buffer, space, ivec(1, 2));`

- `ndwrite_soa<T, N>(data, space, index, value)` - Write a `vec_t<T, N>` to SoA buffer
  * Template parameters: `T` (element type), `N` (vector size)
  * `data: T*` - Pointer to flat buffer
  * `space: index_space_t<S>` - Index space defining spatial dimensions
  * `index: ivec_t<S>` - Multi-dimensional spatial index
  * `value: vec_t<T, N>` - Vector to write
  * Scatters vector components into memory with same layout as `ndread_soa`
  * Usage: `ndwrite_soa<double, 3>(buffer, space, ivec(1, 2), dvec(1.0, 2.0, 3.0));`

# Driver

The `mist-driver.hpp` provides a generic time-stepping driver for physics simulations. It manages the main loop, adaptive time-stepping, and scheduled outputs.

## Physics Concept

Physics modules must satisfy the `Physics` concept by providing:

**Required types:**
- `config_t` - Runtime configuration (grid size, physical parameters, etc.)
- `state_t` - Conservative state variables (density, momentum, energy, etc.). Must implement `fields()`.
- `product_t` - Derived diagnostic quantities (velocity, pressure, etc.). Must implement `fields()`.

**Required functions:**
- `initial_state(config_t) -> state_t` - Generate initial conditions
- `euler_step(config_t, state_t, dt) -> state_t` - Single forward Euler step
- `courant_time(config_t, state_t) -> double` - Maximum stable timestep
- `average(state_t, state_t, alpha) -> state_t` - Convex combination: `(1-alpha)*s0 + alpha*s1`
- `get_product(config_t, state_t) -> product_t` - Compute derived quantities
- `get_time(state_t, kind) -> double` - Extract time from state
  - `kind=0`: Raw simulation time (same units as `courant_time`)
  - `kind=1, 2, ...`: Additional time variables (e.g., orbital phase, forward shock radius)
  - Physics modules define one or more time kinds based on their needs
  - **All time kinds must increase monotonically** during simulation
  - **Must throw `std::out_of_range`** if `kind` is not valid for this physics module
- `zone_count(state_t) -> std::size_t` - Number of computational zones (for performance metrics)
- `timeseries_sample(config_t, state_t) -> std::vector<std::pair<std::string, double>>` - Compute timeseries measurements
  - Returns a vector of (column_name, value) pairs
  - Column names define the available timeseries measurements
  - Order is preserved as defined by physics module
  - Example: `{{"total_mass", 1.0}, {"total_energy", 2.5}, {"max_density", 10.3}}`

## Time Integrators

The driver provides Strong Stability Preserving (SSP) Runge-Kutta methods:
- `rk1_step` - Forward Euler (1st order)
- `rk2_step` - SSP-RK2 (2nd order)
- `rk3_step` - SSP-RK3 (3rd order)

Selected via `driver::config_t::rk_order` (1, 2, or 3).

## Driver State

For restarts to work correctly, the driver maintains internal state that must be persisted alongside the physics state in checkpoint files.

**Scheduled Output State:**
Each output type (message, checkpoint, products, timeseries) has:
```cpp
struct scheduled_output_state {
    int count = 0;        // Number of outputs emitted
    double next_time = 0; // Next scheduled output time
};
```

**Driver State (`driver::state_t`):**
```cpp
struct state_t {
    int iteration = 0;
    
    scheduled_output_state message_state;
    scheduled_output_state checkpoint_state;
    scheduled_output_state products_state;
    scheduled_output_state timeseries_state;
    
    timeseries_t timeseries;  // Accumulated timeseries data
};
```

**Timeseries Data (`driver::timeseries_t`):**
```cpp
struct timeseries_t {
    std::vector<std::pair<std::string, std::vector<double>>> data;
};
```
- Structure: vector of (column_name, values) pairs
- Each column has a name (string) and all samples for that column (vector<double>)
- Column names and order are defined by the physics module's `timeseries_sample()` function
- When a new sample is taken, one value is appended to each column's vector
- Persisted across sessions so timeseries data accumulates throughout the entire run

**Session State (not persisted, lifetime of executable):**
- `double last_message_wall_time` - Wall-clock time of last message (for Mzps calculation between messages)

**Checkpoint Structure:**
When writing checkpoints, the callback receives:
- Physics `state_t` - from the physics module
- Driver state - from the driver (already updated to reflect this output)

The driver state written to the checkpoint reflects the state **after** the output has occurred:
- `checkpoint_count` has been incremented (this checkpoint's number)
- `next_checkpoint_time` has been advanced (when the next checkpoint will occur)

This ensures that restarting from a checkpoint continues correctly without re-executing the same output.

**Terminology:**
- **Run**: A simulation that may have been stopped and restarted any number of times
- **Session**: A single execution of the program (lifetime of executable)
- A run may consist of multiple sessions via checkpointing and restart

**Scheduled Output Execution Order:**
All scheduled outputs follow this uniform sequence:
1. Trigger condition is met (exact or nearest policy)
2. Increment output counter (e.g., `checkpoint_count++`)
3. Advance next scheduled time (e.g., `next_checkpoint_time += interval`)
4. Invoke checkpoint writer with state and updated driver_state

This ensures checkpoint writer always see the "post-output" driver state, which is what gets persisted.

**Restart Behavior:**
When restarting from a checkpoint:
- Driver state is restored from the checkpoint file
- Output counters continue from saved values (e.g., next checkpoint is `chkpt.0005.h5` if `checkpoint_count = 4`)
- Scheduled output times continue from saved values
- Iteration count continues from saved value
- Session state is initialized fresh (e.g., wall-clock timing resets for new session)

## Configuration Structure

The driver uses a two-level configuration structure separating driver settings from physics settings:

**Scheduled Output Config:**
Each output type (message, checkpoint, products, timeseries) has:
```cpp
struct scheduled_output_config {
    double interval = 1.0;
    int interval_kind = 0;
    std::string scheduling = "nearest";  // "nearest" or "exact"
};
```

**Driver Config (`driver::config_t`):**
```cpp
struct config_t {
    int rk_order = 2;
    double cfl = 0.4;
    double t_final = 1.0;
    int max_iter = -1;
    std::string output_format = "ascii";  // "ascii" or "binary"
    
    scheduled_output_config message{0.1, 0, "nearest"};
    scheduled_output_config checkpoint{1.0, 0, "nearest"};
    scheduled_output_config products{0.1, 0, "exact"};
    scheduled_output_config timeseries{0.01, 0, "exact"};
};
```

**Combined Config:**
```cpp
template<Physics P>
struct config {
    driver::config_t driver;
    typename P::config_t physics;
};
```

**Config file format:**
```
config {
    driver {
        rk_order = 3
        cfl = 0.5
        t_final = 2.0
        output_format = "ascii"
        message {
            interval = 0.1
            interval_kind = 0
            scheduling = "nearest"
        }
        checkpoint {
            interval = 1.0
            interval_kind = 0
            scheduling = "nearest"
        }
        products {
            interval = 0.1
            interval_kind = 0
            scheduling = "exact"
        }
        timeseries {
            interval = 0.01
            interval_kind = 0
            scheduling = "exact"
        }
    }
    physics {
        // Physics-specific settings
    }
}
```

## Scheduled Outputs

The driver manages four types of scheduled output. Each output type can be scheduled based on any time kind defined by the physics module (via `get_time(state, kind)`), allowing outputs to recur at (semi-)regular intervals of different time variables.

**Time Kind Conventions:**
- `kind=0`: Raw simulation time (same units as `courant_time` returns)
- `kind=1+`: Physics-specific time variables (e.g. orbital phase, forward shock radius)

**Scheduling Policies:**

Each scheduled output has a policy determining how the driver hits output times:

- **Nearest**: Output occurs at the nearest iteration after the scheduled time
  - Driver checks after each step: if `time >= next_output`, trigger output and advance schedule
  - Output times may drift slightly past exact multiples of `dt_output`
  - Lower overhead, doesn't constrain timestep
  - Use for: checkpoints, diagnostics where exact timing isn't critical

- **Exact**: Driver generates output at precisely the scheduled time
  - When `time < next_output < time + dt`, creates a throw-away state advanced exactly to `next_output`
  - The throw-away state is used only for output; simulation continues from the original state
  - Guarantees outputs at exact multiples: `t = 0, dt_output, 2*dt_output, ...`
  - **Requires `time_kind = 0`** (raw simulation time) - cannot use exact scheduling with other time kinds
  - Higher overhead (extra RK step per output), but ensures exact timing
  - Use for: products, timeseries where exact timing is needed for analysis

Each scheduled output specifies an interval, a time kind, and a scheduling policy.

### 1. Iteration Messages (Console/Logging)
**Purpose:** Lightweight progress monitoring with performance metrics  
**Trigger:** Any time kind  
**Content:** Compact status (iteration, times, timestep, performance)  
**Scheduling:** Configured via `driver::config_t::message` (`scheduled_output_config`)

**Performance Measurement:**
- Driver measures **wall-clock time** between iteration messages
- Reports performance in **Mzps** (million zone-updates per second)
- Calculation: `Mzps = (iterations × zone_count(state)) / (wall_seconds × 1e6)`
- Requires `zone_count(state)` from physics interface
- Messages are written **after** timestep, so first message includes valid Mzps

**Message Format:**
```
[001234] t=3.14159 (1:0.5000 2:0.1233) Mzps=170.123
```
- `[001234]` - 6-digit zero-padded iteration number
- `t=3.14159` - Raw simulation time (kind=0) with 5 decimal places
- `(1:0.5000 2:0.1233)` - Additional time kinds with 4 decimal places
- `Mzps=170.123` - Performance metric with 3 decimal places

**Output Function:** Driver calls `write_iteration_message(message_string)`
- Default implementation prints to stdout
- User can override by defining their own implementation

### 2. Checkpoints (State Persistence)
**Purpose:** Save full simulation state for restart/recovery  
**Trigger:** Any time kind  
**Content:** Complete simulation state for restart  
**Scheduling:** Configured via `driver::config_t::checkpoint` (`scheduled_output_config`)

**Checkpoint file structure:**
```
checkpoint {
    driver_config { ... }   // Full driver configuration
    driver_state { ... }    // Driver state including timeseries data
    physics_config { ... }  // Full physics configuration
    physics_state { ... }   // Physics state variables
}
```

**Output Function:** Driver creates writer and calls `write_checkpoint()`
- Driver constructs filename: `chkpt.{:04d}.{ext}` where ext is "dat" or "bin"
- Driver serializes `driver_config`, `driver_state`, `physics_config`, and `physics_state`

**Output numbering:** 
- Initial: `chkpt.0000{ext}` (written at simulation start)
- Next: `chkpt.0001{ext}`, `chkpt.0002{ext}`, etc.
- Number is checkpoint count, not iteration number
- Extension determined by archive format (e.g., `.h5`, `.dat`, `.bin`)

### 3. Product Files (Derived Quantities)
**Purpose:** Write derived/diagnostic quantities for analysis  
**Trigger:** Any time kind  
**Content:** `product_t` from `get_product(cfg, state)`  
**Scheduling:** Configured via `driver::config_t::products` (`scheduled_output_config`)

**Output Function:** Driver creates writer and calls `write_products()`
- Driver constructs filename: `prods.{:04d}.{ext}` where ext is "dat" or "bin"
- Driver computes `product` via `get_product(physics_config, state)`
- Driver serializes `product` using `serialize()`

**Output numbering:**
- Initial: `prods.0000{ext}` (written at `t=0`)
- Next: `prods.0001{ext}`, `prods.0002{ext}`, etc.
- Number is product output count, not iteration number
- Extension determined by archive format (e.g., `.h5`, `.dat`, `.bin`)

### 4. Timeseries Data (Scalar Diagnostics)
**Purpose:** Record scalar diagnostics over time (total energy, mass, extrema, etc.)  
**Trigger:** Any time kind  
**Content:** User-defined scalar measurements (all type `double`)  
**Scheduling:** Configured via `driver::config_t::timeseries` (`scheduled_output_config`)

**Data Collection:**
- Driver calls `timeseries_sample(physics_config, state)` from physics module
- Returns `std::vector<std::pair<std::string, double>>` with (column_name, value) pairs for this sample
- For each (name, value) pair:
  - If column `name` exists in `driver_state.timeseries.data`: append `value` to that column's vector
  - If column `name` is new: create new column with `value` as first entry
- Column names do not need to be consistent across samples (columns can be added dynamically)
- All measurements are accumulated in `driver_state.timeseries`
- Data persists across sessions (saved in checkpoints)

## Timestep and Termination

The driver uses adaptive timesteps from `courant_time(cfg, state)` multiplied by the CFL factor. The simulation terminates when:
1. `t >= t_final`, or
2. `iteration >= max_iter` (if `max_iter > 0`)

The timestep is never adjusted to hit `t_final` exactly - the simulation simply stops when the termination condition is met. Exact-policy outputs must use `time_kind = 0`.

# Serialization

The `mist/serialize.hpp` provides a lightweight, modular serialization framework for writing and reading simulation data. The framework is format-agnostic and supports ASCII, binary, and HDF5 output through a unified interface.

## Design Philosophy

- **Format-agnostic**: Core serialization logic is independent of output format
- **Type-driven**: Uses C++20 concepts to dispatch based on type traits
- **Composable**: Complex types serialize recursively through their components
- **Strict validation**: All fields must be present during deserialization (no optional fields)
- **Consistent with Mist patterns**: Free functions, public underscore-prefixed members

## Core Interface

Two primary entry points:

```cpp
// Serialize object to archive
template<Archive A, typename T>
void serialize(A& archive, const char* name, const T& obj);

// Deserialize object from archive
template<Archive A, typename T>
void deserialize(A& archive, const char* name, T& obj);
```

## Archive Concept

Archives must implement separate reader and writer interfaces:

```cpp
template<typename A>
concept ArchiveWriter = requires(A& ar, const char* name) {
    { ar.write_scalar(name, int{}) } -> std::same_as<void>;
    { ar.write_scalar(name, double{}) } -> std::same_as<void>;
    { ar.write_string(name, std::string{}) } -> std::same_as<void>;
    { ar.write_array(name, vec_t<double, 3>{}) } -> std::same_as<void>;
    { ar.write_array(name, std::vector<double>{}) } -> std::same_as<void>;
    { ar.begin_group(name) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;  // anonymous group
    { ar.end_group() } -> std::same_as<void>;
};

template<typename A>
concept ArchiveReader = requires(A& ar, const char* name, int& i, double& d) {
    { ar.read_scalar(name, i) } -> std::same_as<void>;
    { ar.read_scalar(name, d) } -> std::same_as<void>;
    { ar.read_string(name, std::string&) } -> std::same_as<void>;
    { ar.read_array(name, vec_t<double, 3>&) } -> std::same_as<void>;
    { ar.read_array(name, std::vector<double>&) } -> std::same_as<void>;
    { ar.begin_group(name) } -> std::same_as<void>;
    { ar.begin_group() } -> std::same_as<void>;  // anonymous group
    { ar.end_group() } -> std::same_as<void>;
    { ar.count_groups(name) } -> std::same_as<std::size_t>;
};
```

**Archive implementations:**
- `ascii_writer` / `ascii_reader` - Human-readable text format
- `binary_writer` / `binary_reader` - Compact binary format (planned)
- `hdf5_writer` / `hdf5_reader` - HDF5 hierarchical data format (planned)

## Serializable Types

The framework automatically handles:

1. **Scalars**: `int`, `float`, `double`, and other arithmetic types
2. **Strings**: `std::string` (quoted with escape sequences)
3. **Static vectors**: `vec_t<T, N>` where `T` is arithmetic
4. **Dynamic vectors**: `std::vector<T>` where `T` is serializable
5. **User-defined types**: Any type with `fields()` method

## Making Types Serializable

Define both const and non-const versions of `fields()`:

```cpp
struct particle_t {
    vec_t<double, 3> position;
    vec_t<double, 3> velocity;
    double mass;
    
    auto fields() const {
        return std::make_tuple(
            field("position", position),
            field("velocity", velocity),
            field("mass", mass)
        );
    }
    
    auto fields() {
        return std::make_tuple(
            field("position", position),
            field("velocity", velocity),
            field("mass", mass)
        );
    }
};
```

The const version is used for writing, the non-const version for reading.

## ASCII Format Specification

The ASCII archive produces human-readable output with the following formatting rules:

### Formatting Rules

1. **Scalars**: `name = value`
   ```
   time = 1.234
   iteration = 42
   ```

2. **Strings**: `name = "value"` (with escape sequences `\"`, `\\`, `\n`, `\t`, `\r`)
   ```
   title = "Blast Wave Simulation"
   path = "output/data"
   ```

3. **Static vectors** (`vec_t<T, N>`): Inline comma-separated arrays
   ```
   position = [0.1, 0.2, 0.15]
   velocity = [1.5, -0.3, 0.0]
   ```

4. **Dynamic vectors of scalars** (`std::vector<T>` where `T` is arithmetic): Inline comma-separated arrays
   ```
   scalar_field = [300.0, 305.2, 298.5, 302.1]
   ```

5. **Dynamic vectors of compounds** (`std::vector<T>` where `T` is user-defined): Multi-line blocks
   ```
   particles {
       {
           position = [0.1, 0.2, 0.15]
           velocity = [1.5, -0.3, 0.0]
           density = 1.2
           pressure = 101325.0
       }
       {
           position = [0.8, 0.7, 0.25]
           velocity = [-0.5, 0.8, 0.2]
           density = 1.1
           pressure = 98000.0
       }
   }
   ```

6. **Nested structures**: Multi-line with indentation
   ```
   grid {
       resolution = [64, 64, 32]
       domain_min = [0.0, 0.0, 0.0]
       domain_max = [1.0, 1.0, 0.5]
   }
   ```

### Delimiter Rules

- **Commas**: Used inside `[ ]` brackets for array elements
- **Newlines**: Used inside `{ }` braces for fields and blocks
- **No commas**: Between struct blocks or field definitions
- **Indentation**: Each nesting level adds one indentation level

### Complete Example

```cpp
struct grid_config_t {
    vec_t<int, 3> resolution;
    vec_t<double, 3> domain_min;
    vec_t<double, 3> domain_max;
    
    auto fields() const {
        return std::make_tuple(
            field("resolution", resolution),
            field("domain_min", domain_min),
            field("domain_max", domain_max)
        );
    }
    
    auto fields() {
        return std::make_tuple(
            field("resolution", resolution),
            field("domain_min", domain_min),
            field("domain_max", domain_max)
        );
    }
};

struct simulation_state_t {
    double time;
    int iteration;
    grid_config_t grid;
    std::vector<particle_t> particles;
    std::vector<double> scalar_field;

    auto fields() const {
        return std::make_tuple(
            field("time", time),
            field("iteration", iteration),
            field("grid", grid),
            field("particles", particles),
            field("scalar_field", scalar_field)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("time", time),
            field("iteration", iteration),
            field("grid", grid),
            field("particles", particles),
            field("scalar_field", scalar_field)
        );
    }
};

// Serialize to ASCII
simulation_state_t state{...};
ascii_writer ar(std::cout);
serialize(ar, "simulation_state", state);
```

**Output:**
```
simulation_state {
    time = 1.234
    iteration = 42
    grid {
        resolution = [64, 64, 32]
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 0.5]
    }
    particles {
        {
            position = [0.1, 0.2, 0.15]
            velocity = [1.5, -0.3, 0.0]
            density = 1.2
            pressure = 101325.0
        }
        {
            position = [0.8, 0.7, 0.25]
            velocity = [-0.5, 0.8, 0.2]
            density = 1.1
            pressure = 98000.0
        }
    }
    scalar_field = [300.0, 305.2, 298.5, 302.1]
}
```

## Deserialization

Deserialization is strict - all fields defined in `fields()` must be present in the input:

```cpp
simulation_state_t state;
ascii_reader ar(std::ifstream("state.dat"));
deserialize(ar, "simulation_state", state);
// Throws exception if any field is missing
```

**Error handling:**
- Missing field: `Error: Field 'velocity' not found in group 'particles/0'`
- Missing group: `Error: Field 'grid' not found in group 'simulation_state'`

## Usage with Different Archive Types

**ASCII (human-readable):**
```cpp
// Writing
ascii_writer aw(std::ofstream("state.dat"));
serialize(aw, "state", state);

// Reading
ascii_reader ar(std::ifstream("state.dat"));
deserialize(ar, "state", state);
```

**Binary (compact):**
```cpp
// Writing
binary_writer bw(std::ofstream("state.bin", std::ios::binary));
serialize(bw, "state", state);

// Reading
binary_reader br(std::ifstream("state.bin", std::ios::binary));
deserialize(br, "state", state);
```

**HDF5 (hierarchical):**
```cpp
// Writing
hdf5_writer hw("state.h5");
serialize(hw, "state", state);

// Reading
hdf5_reader hr("state.h5");
deserialize(hr, "state", state);
```

All three formats use the same `serialize()` / `deserialize()` interface - only the archive type changes.

## Archive Format Traits

For integration with the driver library, archive formats are defined via trait structs that provide type information and factory functions:

```cpp
struct hdf5_t {
    using reader = hdf5_reader;
    using writer = hdf5_writer;
    
    static constexpr const char* extension = ".h5";
    
    static writer make_writer(const std::string& filename);
    static reader make_reader(const std::string& filename);
};

struct ascii_t {
    using reader = ascii_reader;
    using writer = ascii_writer;

    static constexpr const char* extension = ".dat";

    static writer make_writer(const std::string& filename);
    static reader make_reader(const std::string& filename);
};

struct binary_t {
    using reader = binary_reader;
    using writer = binary_writer;

    static constexpr const char* extension = ".bin";

    static writer make_writer(const std::string& filename);
    static reader make_reader(const std::string& filename);
};
```

These traits allow the driver to be parameterized by archive format:

```cpp
// Driver automatically uses correct file extensions and constructs archives
template<typename Archive, Physics P>
void run(const config<P>& cfg, driver_state_t& state) {
    // Driver creates: chkpt.0000.h5, chkpt.0001.h5, etc.
}

// User selects format via template parameter
run<hdf5_t>(cfg, state);   // HDF5 format
run<ascii_t>(cfg, state);  // ASCII format
run<binary_t>(cfg, state); // Binary format
```

The trait struct provides:
- **Type aliases**: `reader` and `writer` types for this format
- **File extension**: String literal for output filenames (e.g., ".h5", ".dat", ".bin")
- **Factory functions**: Construct reader/writer instances from filenames
