# Mist Simulation DSL Design

## Overview

The `simulation<PatchType>` class provides a fluent, domain-specific language for configuring and running distributed simulations. It separates concerns:

1. **Decomposition**: How the domain is split across patches and ranks
2. **Execution**: Which stages to run (exchange, compute, reduce)
3. **Products**: Named outputs for driver integration

The design is generic enough to work with:
- Different dimensionalities (1D, 2D, 3D)
- AMR (Adaptive Mesh Refinement) with multiple patches per rank
- Custom decomposition strategies
- Flexible output extraction

## Architecture

```
simulation<PatchType>
├── patches: std::vector<PatchType>          # Local patches for this rank
├── decomposition methods
│   ├── decompose_cartesian(layout, shape)   # Regular grid decomposition
│   └── decompose_custom(lambda)              # User-defined decomposition
├── stage methods (for fluent API chaining)
│   ├── exchange(stage)
│   ├── compute(stage)
│   └── reduce(stage)
├── product methods
│   ├── define_product(name, extractor)      # Register named output
│   └── run()                                 # Execute & extract all products
└── execution
    └── execute(exchange, compute, reduce)    # Run stages with pipeline
```

## Usage Patterns

### 1. Simple 2D Case (64×64 grid, 4 patches)

```cpp
auto sim = simulation<patch_2d_t>(rank, size, comm)
    .decompose_cartesian(uvec(2, 2), uvec(64, 64))  // 2×2 patch layout
    .exchange(ghost_exchange_stage)
    .compute(compute_stage)
    .reduce(error_reduce_stage)
    .define_product("l2_error", [](const auto& patches) {
        return patches[0].l2_error;
    })
    .define_product("solution_field", [](const auto& patches) {
        return patches[0].u;  // Array view or copy
    });

auto results = sim.run();
double error = std::any_cast<double>(results["l2_error"]);
```

### 2. 1D Wave Equation (1000 points, 10 patches per rank)

```cpp
struct patch_1d_t {
    index_space_t<1> interior;
    array_t<double, 1> u, u_new;
    array_t<double, 1> du_dt;
    double dx;
};

auto sim = simulation<patch_1d_t>(rank, size, comm)
    .decompose_cartesian(uvec(10 * size), uvec(1000))  // 10 patches per rank
    .exchange(wave_ghost_exchange)
    .compute(wave_compute_stage)
    .reduce(wave_energy_reduce_stage)
    .define_product("kinetic_energy", [](const auto& patches) {
        double e = 0.0;
        for (const auto& p : patches) {
            // Accumulate energy from all patches on this rank
        }
        return e;
    });

auto results = sim.run();
double energy = std::any_cast<double>(results["kinetic_energy"]);
```

### 3. 3D Hydrodynamics (256³ grid, 4³ = 64 patches)

```cpp
struct patch_3d_t {
    index_space_t<3> interior;
    array_t<double, 3> rho, vx, vy, vz, e;
    double dx, dy, dz;
};

auto sim = simulation<patch_3d_t>(rank, size, comm)
    .decompose_cartesian(uvec(4, 4, 4), uvec(256, 256, 256))
    .exchange(hydro_ghost_exchange_3d)
    .compute(hydro_flux_compute_3d)
    .reduce(hydro_max_speed_reduce)
    .define_product("max_speed", [](const auto& patches) {
        // Find max speed across all patches
    })
    .define_product("density_field", [](const auto& patches) {
        // Return or copy density array
    });

auto results = sim.run();
double dt = 0.001 / std::any_cast<double>(results["max_speed"]);
```

### 4. AMR with Custom Decomposition (Variable grid structure)

```cpp
struct patch_amr_t {
    index_space_t<2> interior;
    int refinement_level;
    std::vector<array_t<double, 2>> coarse, fine;
    // ... additional fields
};

auto custom_amr_decomp = [](int rank, int size) -> std::vector<patch_amr_t> {
    auto patches = std::vector<patch_amr_t>{};

    // Rank 0: 1 coarse patch at level 0
    if (rank == 0) {
        patches.push_back({.refinement_level = 0, ...});
    }
    // Rank 1-3: 3 fine patches at level 1 (each has 4x resolution)
    else {
        patches.push_back({.refinement_level = 1, ...});
        patches.push_back({.refinement_level = 1, ...});
        patches.push_back({.refinement_level = 1, ...});
    }
    return patches;
};

auto sim = simulation<patch_amr_t>(rank, size, comm)
    .decompose_custom(custom_amr_decomp)  // Different patches per rank
    .exchange(amr_ghost_exchange)
    .compute(amr_compute)
    .reduce(amr_error_reduce)
    .define_product("coarse_solution", [](const auto& patches) {
        // Extract level-0 data
    })
    .define_product("fine_solution", [](const auto& patches) {
        // Extract level-1 data
    })
    .define_product("refinement_efficiency", [](const auto& patches) {
        // Compute ratio of fine to coarse zones
    });

auto results = sim.run();
```

### 5. Multi-Product Output for Driver Framework

```cpp
auto sim = simulation<patch_t>(rank, size, comm)
    .decompose_cartesian(uvec(4, 4), uvec(256, 256))
    .exchange(exchange_stage)
    .compute(compute_stage)
    .reduce(error_stage)
    .define_product("solution", [](const auto& patches) {
        // Gather solution from all patches
    })
    .define_product("l2_error", [](const auto& patches) {
        return patches[0].l2_error;
    })
    .define_product("iteration_stats", [](const auto& patches) {
        struct Stats {
            int num_patches;
            double avg_time;
        };
        return Stats{...};
    })
    .define_product("grid_spacing", [](const auto& patches) {
        return patches[0].dx;
    });

auto results = sim.run();

// Access products with type-safe extraction
auto solution = std::any_cast<field_t>(results["solution"]);
auto error = std::any_cast<double>(results["l2_error"]);
auto stats = std::any_cast<Stats>(results["iteration_stats"]);
```

## Key Design Principles

### 1. Separation of Concerns

- **Decomposition** (how domain is partitioned) is separate from
- **Stages** (what computation to perform) is separate from
- **Products** (what to extract/output)

This allows:
- Changing decomposition without rewriting stages
- Adding new products without recomputing
- Testing stages independently

### 2. Genericity

The `simulation<PatchType>` template works with any patch structure:
- 1D/2D/3D determined by `index_space_t<Rank>`
- AMR handled by custom decomposition returning multiple patches
- Different field layouts supported via PatchType struct definition

### 3. Driver Integration

Named products (`define_product`) avoid global state and enable:
- Clean separation between simulation and driver/framework
- Multiple outputs from one simulation run
- Type-safe extraction via `std::any_cast`

### 4. Fluent API

Method chaining (returning `simulation&`) provides:
- Readable, declarative configuration
- Single point of setup before execution
- Natural flow: decompose → define stages → register products → run

## Implementation Notes

### Decomposition Strategies

**Cartesian** (most common):
```cpp
decompose_cartesian(uvec(2, 2, 2), uvec(64, 64, 64))
// Creates 2×2×2 = 8 patches on global 64³ domain
// Distributes patches round-robin across ranks
```

**Custom** (for AMR, load balancing, etc.):
```cpp
decompose_custom([](int rank, int size) {
    // Returns std::vector<PatchType> for this rank
    // Patches can vary in count/size per rank
    // Enables AMR, load balancing, irregular domains
})
```

### Execution Model

```cpp
// 1. Decomposition creates patches_ (owned by this rank)
sim.decompose_cartesian(...);  // patches_ now has local patches

// 2. Execute pipeline with stages
sim.execute(exchange_stage, compute_stage, reduce_stage);
// Internally uses:
//   auto pipe = parallel::pipeline(exchange, compute, reduce);
//   parallel::execute(pipe, patches_, comm_, scheduler, profiler);

// 3. Define extractors (don't execute yet)
sim.define_product("result1", [](const auto& patches) { ... });

// 4. Run() actually invokes extractors
auto results = sim.run();  // products extracted from patches_
```

### Product Extraction

Products are defined as lambdas that operate on the **final state** of patches:

```cpp
.define_product("error", [](const std::vector<patch_t>& patches) {
    // At this point: compute and reduce stages have completed
    // patches[i].l2_error has been finalized by reduce stage
    return patches[0].l2_error;
})
```

This works because:
1. Stages modify patches in-place
2. Products are extracted after all stages complete
3. Multiple products share the same final state (no recomputation)

## Future Enhancements

1. **Stage Storage**: Store stages in DSL, defer execution
   ```cpp
   auto sim = simulation<patch_t>(...)
       .exchange(stage)   // Store stage, don't execute yet
       .compute(stage)
       .reduce(stage)
       .run();            // Execute all at once
   ```

2. **Pipeline Caching**: For repeated runs with same topology
   ```cpp
   auto cached_plan = sim.cached_exchange_plan();
   // Reuse plan across iterations
   ```

3. **Measurement Stages**: Inline diagnostics
   ```cpp
   .measure("energy", [](const auto& p) { return kinetic_energy(p); })
   .measure("vorticity", [](const auto& p) { return compute_vorticity(p); })
   ```

4. **AMR-Specific Helpers**: Refinement logic
   ```cpp
   .amr_refine([](const auto& patches) { /* mark cells */ })
   .amr_coarsen([](const auto& patches) { /* unmark cells */ })
   ```

## Migration Path

### From existing code:

**Old pattern:**
```cpp
auto cfg = config_t{};
auto patches = create_patches(cfg, rank, size);
auto pipe = parallel::pipeline(exchange, compute, reduce);
auto sched = parallel::scheduler_t{};
parallel::execute(pipe, patches, comm, sched, profiler);
double result = patches[0].l2_error;
```

**New pattern:**
```cpp
auto results = simulation<patch_t>(rank, size, comm)
    .decompose_cartesian(uvec(2, 2), uvec(64, 64))
    .execute(exchange, compute, reduce)
    .define_product("l2_error", [](const auto& p) { return p[0].l2_error; })
    .run();
double result = std::any_cast<double>(results["l2_error"]);
```

## See Also

- `examples/laplacian2d/laplacian2d.cpp`: Working examples with main_a, main_b, main_c, main_d
- `include/mist/pipeline.hpp`: Underlying stage execution infrastructure
- `include/mist/comm.hpp`: Communication layer for exchange/reduce
