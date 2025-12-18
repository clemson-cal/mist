# Driver

The driver library (`mist/driver.hpp`) provides an interactive time-stepping framework for physics simulations.

## Overview

The driver is a REPL (read-eval-print loop) that responds to commands for advancing time, configuring physics, and generating outputs. It supports interactive use, scripted batch runs, and programmatic control via socket.

## Physics Concept

Physics modules must satisfy the `Physics` concept:

**Required types:**
- `config_t` — runtime configuration (must have `fields()`)
- `initial_t` — initial condition parameters (must have `fields()`)
- `state_t` — conservative state (must have `fields()`)
- `product_t` — derived diagnostic quantities
- `exec_context_t` — execution context (config, initial, runtime data)

**Required functions:**
```cpp
default_physics_config(std::type_identity<P>) -> config_t
default_initial_config(std::type_identity<P>) -> initial_t
initial_state(exec_context_t) -> state_t
advance(state_t&, exec_context_t, double dt_max) -> void
courant_time(state_t, exec_context_t) -> double
zone_count(state_t, exec_context_t) -> size_t
names_of_time(std::type_identity<P>) -> vector<string>
names_of_timeseries(std::type_identity<P>) -> vector<string>
names_of_products(std::type_identity<P>) -> vector<string>
get_time(state_t, string) -> double
get_timeseries(state_t, string, exec_context_t) -> double
get_product(state_t, string, exec_context_t) -> product_t
```

## Program Structure

```cpp
int main() {
    auto physics = mist::driver::make_physics<my_physics>();
    auto state = mist::driver::state_t{};
    auto engine = mist::driver::engine_t{state, *physics};
    auto session = mist::driver::repl_session_t{engine};
    session.run();
}
```

For socket mode (used by Python interface and TUI):
```cpp
if (use_socket) {
    auto session = mist::driver::socket_session_t{engine};
    session.run();
}
```

## Interactive Commands

### Stepping
```
n++              # advance 1 iteration
n += 10          # advance 10 iterations
n -> 1000        # advance to iteration 1000
t += 0.1         # advance time by 0.1
t -> 1.0         # advance to time 1.0
orbit += 3.0     # advance until orbit increases by 3.0
orbit -> 60.0    # advance until orbit reaches 60.0
```

### Configuration
```
set output=ascii              # output format: ascii|binary|hdf5
set physics key=val           # set physics config
set initial key=val           # set initial config (before init only)
set exec num_threads=4        # set execution config
```

### Selection
```
select products               # select all products
select products rho vel       # select specific products
select timeseries t mass      # select timeseries columns
```

### Sampling
```
do timeseries                 # record one sample
```

### File I/O
```
write checkpoint              # auto-named: chkpt.NNNN.dat
write checkpoint myfile.dat   # explicit filename
write products                # auto-named: prods.NNNN.dat
write timeseries data.dat     # explicit filename
```

### Repeating Commands
```
repeat 1 n show iteration     # show status every iteration
repeat 0.1 t do timeseries    # sample every 0.1 time units
repeat 10 n write checkpoint  # checkpoint every 10 iterations
repeat list                   # list active repeats
repeat clear                  # clear all repeats
```

### State Management
```
init                          # generate initial state
reset                         # clear state, reset driver
load checkpoint.dat           # load checkpoint or config
```

### Information
```
show all                      # everything
show physics                  # physics config
show initial                  # initial config
show products                 # available/selected products
show timeseries               # timeseries data
show profiler                 # timing data
help                          # command help
stop                          # exit (or quit, q)
```

## Example Session

```
$ ./advect1d
> show physics
physics {
    rk_order = 1
    cfl = 0.4
    wavespeed = 1.0
}
> init
[000000] t=0.00000
> select timeseries time total_mass
> n += 5
[000005] t=0.0100 Mzps=3.465
> do timeseries
> t -> 1.0
[000500] t=1.0000 Mzps=2.954
> write checkpoint
Wrote chkpt.0000.dat
> stop
```

## Scripted Runs

Use heredoc for batch execution:
```bash
./advect1d << EOF
set initial num_zones=1000
repeat 1 n show iteration
init
t -> 1.0
write checkpoint
stop
EOF
```

## Output Files

**Checkpoints:** `chkpt.NNNN.dat`
- Full program state (physics config, state, driver state)
- Reload with `load` command

**Products:** `prods.NNNN.dat`
- Derived quantities from `get_product()`

**Timeseries:** User-specified filename
- Accumulated samples from `do timeseries`

File numbering increments with each write command.

## Timeseries Workflow

1. Select columns: `select timeseries col1 col2`
2. Record samples: `do timeseries` (or via repeat)
3. Write to file: `write timeseries data.dat`
4. Clear if needed: `clear timeseries`

Timeseries data persists in checkpoints.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  session_t (repl_session_t / socket_session_t)  │
│  - readline, I/O, command parsing               │
│  - response formatting                          │
└─────────────────────┬───────────────────────────┘
                      │ command_t
                      ▼
┌─────────────────────────────────────────────────┐
│  engine_t                                       │
│  - state machine, no I/O                        │
│  - execute(command_t, callback) → response_t    │
│  - Ctrl-C handling                              │
└─────────────────────┬───────────────────────────┘
                      │ method calls
                      ▼
┌─────────────────────────────────────────────────┐
│  physics_interface_t                            │
│  - type-erased interface                        │
│  - implemented by physics_impl_t<P>             │
└─────────────────────────────────────────────────┘
```

## Internal Types

### state_t
Driver state (persisted in checkpoints):
- `output_format` — ascii/binary/hdf5
- `iteration` — current iteration count
- `checkpoint_count`, `products_count`, `timeseries_count` — file counters
- `timeseries` — accumulated samples
- `repeating_commands` — active repeat commands
- `selected_products` — product selection

### engine_t
Runtime state machine:
- Owns `state_t`
- Reference to `physics_interface_t`
- Command execution via `execute(command_t, callback)`

### physics_interface_t
Type-erased physics operations. Implemented by `physics_impl_t<P>` for physics module P.

---

## Command Reference

### Stepping
| Command | Args | Description |
|---------|------|-------------|
| advance_by | var, delta | Advance until var increases by delta |
| advance_to | var, target | Advance until var reaches target |

### Configuration
| Command | Args | Description |
|---------|------|-------------|
| set_output | format | Set output format |
| set_physics | key, value | Set physics config |
| set_initial | key, value | Set initial config |
| set_exec | key, value | Set execution config |

### Selection
| Command | Args | Description |
|---------|------|-------------|
| select_timeseries | [cols] | Select columns (empty = all) |
| select_products | [prods] | Select products (empty = all) |

### Sampling
| Command | Args | Description |
|---------|------|-------------|
| do_timeseries | — | Record one sample |

### Write
| Command | Args | Description |
|---------|------|-------------|
| write_checkpoint | [file] | Write checkpoint |
| write_products | [file] | Write products |
| write_timeseries | [file] | Write timeseries |
| write_physics | file | Write physics config |
| write_initial | file | Write initial config |
| write_driver | file | Write driver state |
| write_profiler | file | Write profiler data |

### Repeating
| Command | Args | Description |
|---------|------|-------------|
| repeat_add | interval, unit, cmd | Add repeating command |
| clear_repeat | — | Clear all repeats |

### State Management
| Command | Args | Description |
|---------|------|-------------|
| init | — | Initialize physics state |
| reset | — | Clear state |
| load | file | Load checkpoint/config |

### Show
| Command | Args | Description |
|---------|------|-------------|
| show_message | — | Current status |
| show_all | — | Everything |
| show_physics | — | Physics config |
| show_initial | — | Initial config |
| show_timeseries | — | Timeseries info |
| show_products | — | Products info |
| show_profiler | — | Profiler data |
| show_driver | — | Driver state |

### Control
| Command | Args | Description |
|---------|------|-------------|
| help | — | Help text |
| stop | — | Exit |
