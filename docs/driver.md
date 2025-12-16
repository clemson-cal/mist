# Driver Design

## Terminology

- Session — one invocation of the executable
- Run — logical simulation run, can span multiple sessions via checkpoint/resume

## Architecture

```
┌───────────────────────────────────────────────┐
│  session_t                                    │
│  - readline, I/O, REPL                        │
│  - command parsing                            │
│  - response formatting (colors, tables)       │
│  - owns engine_t                              │
└───────────────────┬───────────────────────────┘
                    │ command_t
                    ▼
┌───────────────────────────────────────────────┐
│  engine_t                                     │
│  - pure state machine, no console I/O         │
│  - execute(command_t, callback) → response_t  │
│  - handles Ctrl-C during advance              │
│  - owns state_t + ref to physics_interface_t  │
└───────────────────┬───────────────────────────┘
                    │ method calls
                    ▼
┌───────────────────────────────────────────────┐
│  physics_interface_t                          │
│  - type-erased interface to physics module    │
│  - owns physics config, initial, state        │
│  - advance(), get_time(), get_timeseries()    │
│  - implemented by physics_impl_t<P>           │
└───────────────────────────────────────────────┘
```

## Types

### state_t
Checkpoint-persistent driver state (non-templated):
- output_format format
- int iteration
- int checkpoint_count, products_count, timeseries_count
- map<str, [double]> timeseries
- [repeating_command_t] repeating_commands (pre-parsed, validated)
- [str] selected_products

### engine_t
Runtime state machine:
- Owns state_t
- Reference to physics_interface_t
- Handles command execution via execute(command_t, callback)
- Handles Ctrl-C interruption during advance

### session_t
Interactive session layer:
- Owns engine_t
- Readline integration, command history
- Parses text input → command_t
- Formats response_t → colored console output
- Handles script loading (.prog/.mist files)

### physics_interface_t
Type-erased physics operations:
- time_names(), timeseries_names(), product_names()
- init(), reset(), has_state()
- advance(), get_time(), get_timeseries()
- set_physics(), set_initial(), set_exec()
- write_physics(), write_initial(), write_checkpoint(), etc.
- zone_count(), profiler_data()

### physics_impl_t<P>
Templated implementation of physics_interface_t for physics module P.

---

## Commands

### Stepping
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| advance_by       | str var, double delta       | Advance until var increases by delta |
| advance_to       | str var, double target      | Advance until var reaches target     |

### Configuration
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| set_output       | str format                  | Set output format (ascii/binary/hdf5)|
| set_physics      | str key, str value          | Set physics config key=value         |
| set_initial      | str key, str value          | Set initial config (only before init)|
| set_exec         | str key, str value          | Set exec config (e.g. num_threads)   |

### Selection
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| select_timeseries| [str] cols                  | Select columns (empty = all)         |
| select_products  | [str] prods                 | Select products (empty = all)        |

### Sampling
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| do_timeseries    | —                           | Record one timeseries sample         |

### Write
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| write_physics    | str dest                    | Required filename or "socket"        |
| write_initial    | str dest                    | Required filename or "socket"        |
| write_driver     | str dest                    | Required filename or "socket"        |
| write_profiler   | str dest                    | Required filename or "socket"        |
| write_timeseries | str? dest                   | Optional (auto-generates name)       |
| write_checkpoint | str? dest                   | Optional (auto-generates name)       |
| write_products   | str? dest                   | Optional (auto-generates name)       |

### Repeating
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| repeat_add       | double, str unit, command_t | Add repeating command (pre-parsed)   |
| clear_repeat     | —                           | Clear all repeating commands         |

### State Management
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| init             | —                           | Initialize physics state             |
| reset            | —                           | Clear physics and driver state       |
| load             | str filename                | Load checkpoint or config file       |

### Show
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| show_message     | —                           | Current iteration status             |
| show_all         | —                           | Everything                           |
| show_physics     | —                           | Physics config                       |
| show_initial     | —                           | Initial config                       |
| show_timeseries  | —                           | Timeseries selection + counts        |
| show_products    | —                           | Product selection                    |
| show_profiler    | —                           | Profiler data                        |
| show_driver      | —                           | Driver state                         |

### Control
| Command          | Args                        | Description                          |
|------------------|-----------------------------|--------------------------------------|
| help             | —                           | Help text                            |
| stop             | —                           | Exit                                 |

Total: **31 commands**

---

## Responses

Engine returns responses via callback: execute(command_t, function<void(response_t)>)

### Status
| Response         | Fields                      | From                                 |
|------------------|-----------------------------|--------------------------------------|
| ok               | str message                 | set_*, select_*, init, reset, load   |
| error            | str what                    | any command that fails               |
| interrupted      | —                           | advance_* (Ctrl-C)                   |
| stopped          | —                           | stop                                 |

### Iteration
| Response         | Fields                      | From                                 |
|------------------|-----------------------------|--------------------------------------|
| iteration_status | int n, map times, double dt, double zps | show_message, advance_*  |
| timeseries_sample| map<str,double> values      | do_timeseries                        |

### Show (serialized)
| Response         | Fields                      | From                                 |
|------------------|-----------------------------|--------------------------------------|
| physics_config   | str text                    | show_physics                         |
| initial_config   | str text                    | show_initial                         |
| driver_state     | str text                    | show_driver                          |
| help_text        | str text                    | help                                 |

### Show (structured)
| Response         | Fields                      | From                                 |
|------------------|-----------------------------|--------------------------------------|
| timeseries_info  | [str] available, [str] selected, map counts | show_timeseries      |
| products_info    | [str] available, [str] selected | show_products                    |
| profiler_info    | [entry_t] entries, double total | show_profiler                    |

### Write
| Response         | Fields                      | From                                 |
|------------------|-----------------------------|--------------------------------------|
| wrote_file       | str filename, size_t bytes  | write_* (to file)                    |
| socket_listening | int port                    | write_* socket (waiting)             |
| socket_sent      | size_t bytes                | write_* socket (done)                |
| socket_cancelled | —                           | write_* socket (Ctrl-C)              |

Total: **16 response types**
