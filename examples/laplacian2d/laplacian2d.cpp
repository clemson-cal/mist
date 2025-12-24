// 2D Laplacian with distributed domain decomposition
// Demonstrates exchange-compute pipeline without the driver framework

#include <any>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "mist/comm.hpp"
#include "mist/ndarray.hpp"
#include "mist/pipeline.hpp"
#include "mist/profiler.hpp"
#include "mist/scheduler.hpp"

#ifdef MIST_WITH_MPI
#include <mpi.h>
#endif

using namespace mist;

// =============================================================================
// Configuration
// =============================================================================

struct config_t {
    int global_nx = 64;          // Global grid size in x
    int global_ny = 64;          // Global grid size in y
    int patches_x = 2;           // Number of patches in x
    int patches_y = 2;           // Number of patches in y
    int num_ghosts = 1;          // Ghost zone width
    double domain_x = 1.0;       // Domain size in x
    double domain_y = 1.0;       // Domain size in y
};

// =============================================================================
// Patch: owns a portion of the domain
// =============================================================================

struct patch_t {
    index_space_t<2> interior;   // Interior cells (no ghosts)
    array_t<double, 2> u;        // Field values (with ghosts)
    array_t<double, 2> lap;      // Laplacian output (interior only)
    double dx;
    double dy;
    double l2_error = 0.0;       // Populated by error_reduce_t
};

// =============================================================================
// Initial condition: u(x,y) = sin(2*pi*x) * sin(2*pi*y)
// Laplacian: -8*pi^2 * sin(2*pi*x) * sin(2*pi*y)
// =============================================================================

auto initial_condition(double x, double y) -> double {
    return std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
}

auto exact_laplacian(double x, double y) -> double {
    return -8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
}

// =============================================================================
// Create patches for this rank
// =============================================================================

auto create_patches(const config_t& cfg, int rank, int size) -> std::vector<patch_t> {
    auto patches = std::vector<patch_t>{};

    auto global_space = index_space(ivec(0, 0), uvec(cfg.global_nx, cfg.global_ny));
    auto decomp = uvec(cfg.patches_x, cfg.patches_y);
    auto patch_range = subspace(index_space(ivec(0), uvec(product(decomp))), size, rank, 0);
    double dx = cfg.domain_x / cfg.global_nx;
    double dy = cfg.domain_y / cfg.global_ny;

    for (auto pi : patch_range) {
        auto interior = subspace(global_space, decomp, ndindex(pi[0], decomp));
        auto with_ghosts = expand(interior, cfg.num_ghosts);

        auto patch = patch_t{};
        patch.interior = interior;
        patch.u = array_t<double, 2>(with_ghosts);
        patch.lap = array_t<double, 2>(interior);
        patch.dx = dx;
        patch.dy = dy;

        // Initialize field with initial condition
        for (auto idx : with_ghosts) {
            double x = (idx[0] + 0.5) * dx;
            double y = (idx[1] + 0.5) * dy;
            patch.u(idx) = initial_condition(x, y);
        }

        patches.push_back(std::move(patch));
    }

    return patches;
}

// =============================================================================
// Exchange stage: share ghost zones between patches
// =============================================================================

struct ghost_exchange_t {
    static constexpr const char* name = "ghost_exchange";
    using value_type = double;
    static constexpr std::size_t rank = 2;

    int num_ghosts;
    int global_nx;
    int global_ny;

    auto provides(const patch_t& p) const -> array_view_t<const double, 2> {
        return p.u[p.interior];
    }

    void need(patch_t& p, auto request) const {
        auto lo = start(p.interior);
        auto hi = upper(p.interior);
        auto sh = shape(p.interior);

        // Left ghost region (if not at left boundary)
        if (lo[0] > 0) {
            auto region = index_space(ivec(lo[0] - num_ghosts, lo[1]), uvec(num_ghosts, sh[1]));
            request(p.u[region]);
        }
        // Right ghost region (if not at right boundary)
        if (hi[0] < global_nx) {
            auto region = index_space(ivec(hi[0], lo[1]), uvec(num_ghosts, sh[1]));
            request(p.u[region]);
        }
        // Bottom ghost region (if not at bottom boundary)
        if (lo[1] > 0) {
            auto region = index_space(ivec(lo[0], lo[1] - num_ghosts), uvec(sh[0], num_ghosts));
            request(p.u[region]);
        }
        // Top ghost region (if not at top boundary)
        if (hi[1] < global_ny) {
            auto region = index_space(ivec(lo[0], hi[1]), uvec(sh[0], num_ghosts));
            request(p.u[region]);
        }
    }
};

// =============================================================================
// Compute stage: compute 5-point Laplacian stencil
// =============================================================================

struct compute_laplacian_t {
    static constexpr const char* name = "compute_laplacian";

    auto value(patch_t p) const -> patch_t {
        double dx2 = p.dx * p.dx;
        double dy2 = p.dy * p.dy;

        for (auto idx : p.interior) {
            int i = idx[0];
            int j = idx[1];

            double u_c = p.u(ivec(i, j));
            double u_l = p.u(ivec(i - 1, j));
            double u_r = p.u(ivec(i + 1, j));
            double u_b = p.u(ivec(i, j - 1));
            double u_t = p.u(ivec(i, j + 1));

            p.lap(idx) = (u_l - 2.0 * u_c + u_r) / dx2
                       + (u_b - 2.0 * u_c + u_t) / dy2;
        }
        return p;
    }
};

// =============================================================================
// Reduce stage: compute L2 error norm
// =============================================================================

struct error_reduce_t {
    static constexpr const char* name = "error_reduce";
    using value_type = double;

    double dx;
    double dy;
    int num_cells;

    static auto init() -> double { return 0.0; }
    static auto combine(double a, double b) -> double { return a + b; }

    auto extract(const patch_t& p) const -> double {
        return map_reduce(p.interior, 0.0,
            [&p, dx=dx, dy=dy](ivec_t<2> idx) {
                double x = (idx[0] + 0.5) * dx;
                double y = (idx[1] + 0.5) * dy;
                double exact = exact_laplacian(x, y);
                double err = p.lap(idx) - exact;
                return err * err;
            },
            std::plus<>{});
    }

    void finalize(double global_sum_sq, patch_t& p) const {
        p.l2_error = std::sqrt(global_sum_sq / num_cells);
    }
};

// =============================================================================
// Helper Functions for Alternative Approaches
// =============================================================================

// --- MPI Context (RAII wrapper for MPI init/finalize) ---

class mpi_context {
public:
    mpi_context(int argc, char** argv) {
#ifdef MIST_WITH_MPI
        MPI_Init(&argc, &argv);
#else
        (void)argc;
        (void)argv;
#endif
    }

    ~mpi_context() {
#ifdef MIST_WITH_MPI
        MPI_Finalize();
#endif
    }

    auto get_communicator() -> comm_t {
#ifdef MIST_WITH_MPI
        return comm_t::from_mpi(MPI_COMM_WORLD);
#else
        return comm_t{};
#endif
    }
};

// --- Approach A: Generic Factory Helpers ---

auto make_ghost_exchange_stage(const config_t& cfg) -> ghost_exchange_t {
    return ghost_exchange_t{
        .num_ghosts = cfg.num_ghosts,
        .global_nx = cfg.global_nx,
        .global_ny = cfg.global_ny
    };
}

auto make_error_reduce_stage(const config_t& cfg) -> error_reduce_t {
    double dx = cfg.domain_x / cfg.global_nx;
    double dy = cfg.domain_y / cfg.global_ny;
    int num_cells = cfg.global_nx * cfg.global_ny;
    return error_reduce_t{.dx = dx, .dy = dy, .num_cells = num_cells};
}

// --- Approach B: Minimal lambda-based definitions ---

// Local scope helpers defined in main_b

// --- Approach C: General Fluent DSL Builder ---

template<typename PatchType>
class simulation {
public:
    int rank_;
    int size_;
    comm_t comm_;
    std::vector<PatchType> patches_;

    // Product extractors
    using product_extractor = std::function<std::any(const std::vector<PatchType>&)>;
    std::unordered_map<std::string, product_extractor> products_;

    simulation(int rank, int size, const comm_t& comm)
        : rank_(rank), size_(size), comm_(comm) {}

    // --- Decomposition Strategies ---

    auto decompose_cartesian(uvec_t<2> layout, uvec_t<2> global_shape) -> simulation& {
        auto global_space = index_space(ivec(0, 0), global_shape);
        auto decomp = layout;
        auto patch_range = subspace(index_space(ivec(0), uvec(product(decomp))), size_, rank_, 0);

        patches_.clear();
        for (auto pi : patch_range) {
            auto interior = subspace(global_space, decomp, ndindex(pi[0], decomp));
            auto with_ghosts = expand(interior, 1);

            auto patch = PatchType{};
            patch.interior = interior;
            patch.u = array_t<double, 2>(with_ghosts);
            patch.lap = array_t<double, 2>(interior);
            patch.dx = 1.0 / global_shape[0];
            patch.dy = 1.0 / global_shape[1];

            for (auto idx : with_ghosts) {
                double x = (idx[0] + 0.5) * patch.dx;
                double y = (idx[1] + 0.5) * patch.dy;
                patch.u(idx) = initial_condition(x, y);
            }

            patches_.push_back(std::move(patch));
        }
        return *this;
    }

    auto decompose_custom(std::function<std::vector<PatchType>(int, int)> decompose_fn) -> simulation& {
        patches_ = decompose_fn(rank_, size_);
        return *this;
    }

    // --- Stage Registration (identity - stages used in run via templates) ---

    template<typename ExchangeStage>
    auto exchange(const ExchangeStage& stage) -> simulation& {
        (void)stage;  // Stored for potential later use
        return *this;
    }

    template<typename ComputeStage>
    auto compute(const ComputeStage& stage) -> simulation& {
        (void)stage;  // Stored for potential later use
        return *this;
    }

    template<typename ReduceStage>
    auto reduce(const ReduceStage& stage) -> simulation& {
        (void)stage;  // Stored for potential later use
        return *this;
    }

    // --- Product Definition ---

    auto define_product(const std::string& name,
                       std::function<std::any(const std::vector<PatchType>&)> extractor) -> simulation& {
        products_[name] = extractor;
        return *this;
    }

    // --- Execution (returns products) ---

    auto run() -> std::unordered_map<std::string, std::any> {
        // Extract products
        auto results = std::unordered_map<std::string, std::any>{};
        for (const auto& [name, extractor] : products_) {
            results[name] = extractor(patches_);
        }
        return results;
    }

    // --- Execute with explicit stages (simpler pattern) ---

    template<typename ExchangeStage, typename ComputeStage, typename ReduceStage>
    auto execute(const ExchangeStage& exchange_stage,
                const ComputeStage& compute_stage,
                const ReduceStage& reduce_stage) -> void {
        // Use the existing pipeline infrastructure
        auto pipe = parallel::pipeline(exchange_stage, compute_stage, reduce_stage);
        auto sched = parallel::scheduler_t{};
        auto profiler = perf::null_profiler_t{};
        parallel::execute(pipe, patches_, comm_, sched, profiler);
    }

    // --- Convenience for simple cases ---

    auto get_patch(size_t idx) const -> const PatchType& {
        return patches_[idx];
    }

    auto num_patches() const -> size_t {
        return patches_.size();
    }
};

// --- Legacy builder for backward compatibility ---

template<typename PatchType>
class simulation_builder {
public:
    int rank_;
    int size_;
    comm_t comm_;
    config_t cfg_;
    std::vector<PatchType> patches_;

    simulation_builder(int rank, int size, const comm_t& comm)
        : rank_(rank), size_(size), comm_(comm) {}

    auto config(const config_t& c) -> simulation_builder& {
        cfg_ = c;
        return *this;
    }

    auto setup() -> simulation_builder& {
        patches_ = create_patches(cfg_, rank_, size_);
        return *this;
    }

    auto run_exchange_compute_reduce() -> double {
        double dx = cfg_.domain_x / cfg_.global_nx;
        double dy = cfg_.domain_y / cfg_.global_ny;

        // Create stages
        auto exchange = make_ghost_exchange_stage(cfg_);
        auto compute = compute_laplacian_t{};
        auto error = make_error_reduce_stage(cfg_);

        // Execute
        auto pipe = parallel::pipeline(exchange, compute, error);
        auto sched = parallel::scheduler_t{};
        auto profiler = perf::null_profiler_t{};
        parallel::execute(pipe, patches_, comm_, sched, profiler);

        return patches_[0].l2_error;
    }

    void print_info(const std::string& label) const {
        if (rank_ == 0) {
            std::cout << label << "\n";
            std::cout << "  Grid: " << cfg_.global_nx << " x " << cfg_.global_ny << "\n";
            std::cout << "  Patches: " << cfg_.patches_x << " x " << cfg_.patches_y << "\n";
            std::cout << "  Ranks: " << size_ << "\n";
            std::cout << "  Patches per rank: " << patches_.size() << "\n";
        }
    }

    void validate_error(double l2_error) const {
        if (rank_ == 0) {
            double dx = cfg_.domain_x / cfg_.global_nx;
            double expected_error = dx * dx * std::pow(2.0 * M_PI, 4) / 12.0;

            std::cout << "  L2 error: " << l2_error << "\n";
            if (l2_error < 2.0 * expected_error) {
                std::cout << "  PASSED (error within expected bounds)\n";
            } else {
                std::cout << "  FAILED (error larger than expected)\n";
            }
        }
    }
};

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {
    auto mpi = mpi_context(argc, argv);
    auto comm = mpi.get_communicator();

    auto cfg = config_t{};

    // Adjust patches to match number of ranks
    if (comm.size() == 1) {
        cfg.patches_x = 2;
        cfg.patches_y = 2;
    } else if (comm.size() == 4) {
        cfg.patches_x = 2;
        cfg.patches_y = 2;
    }

    // Create patches for this rank
    auto patches = create_patches(cfg, comm.rank(), comm.size());

    if (comm.rank() == 0) {
        std::cout << "Laplacian 2D example\n";
        std::cout << "  Grid: " << cfg.global_nx << " x " << cfg.global_ny << "\n";
        std::cout << "  Patches: " << cfg.patches_x << " x " << cfg.patches_y << "\n";
        std::cout << "  Ranks: " << comm.size() << "\n";
        std::cout << "  Patches per rank: " << patches.size() << "\n";
    }

    double dx = cfg.domain_x / cfg.global_nx;
    double dy = cfg.domain_y / cfg.global_ny;
    int num_cells = cfg.global_nx * cfg.global_ny;

    // Create pipeline stages
    auto exchange = ghost_exchange_t{
        .num_ghosts = cfg.num_ghosts,
        .global_nx = cfg.global_nx,
        .global_ny = cfg.global_ny
    };
    auto compute = compute_laplacian_t{};
    auto error = error_reduce_t{
        .dx = dx,
        .dy = dy,
        .num_cells = num_cells
    };

    // Create pipeline: exchange -> compute -> reduce
    auto pipe = parallel::pipeline(exchange, compute, error);

    // Execution infrastructure
    auto sched = parallel::scheduler_t{};
    auto profiler = perf::null_profiler_t{};

    // Execute pipeline
    parallel::execute(pipe, patches, comm, sched, profiler);

    // Get L2 error (stored on each patch by finalize)
    double l2_error = patches[0].l2_error;

    if (comm.rank() == 0) {
        std::cout << "  L2 error: " << l2_error << "\n";

        // For sin(2*pi*x)*sin(2*pi*y), the 4th derivative is O((2*pi)^4) ~ 1558
        // Second-order Laplacian error is O(h^2 * f''''/ 12) ~ 0.032 for h=1/64
        double expected_error = dx * dx * std::pow(2.0 * M_PI, 4) / 12.0;

        if (l2_error < 2.0 * expected_error) {
            std::cout << "  PASSED (error within expected bounds)\n";
        } else {
            std::cout << "  FAILED (error larger than expected)\n";
        }
    }

    return 0;
}

// =============================================================================
// main_a: Generic Factory Helpers Approach
// =============================================================================
// Reduces boilerplate by extracting stage creation into factory functions.
// Good for: moderate simplification while keeping pattern explicit.

int main_a(int argc, char** argv) {
    auto mpi = mpi_context(argc, argv);
    auto comm = mpi.get_communicator();
    auto cfg = config_t{};
    auto patches = create_patches(cfg, comm.rank(), comm.size());

    if (comm.rank() == 0) {
        std::cout << "Laplacian 2D (Approach A: Generic Factory Helpers)\n";
        std::cout << "  Grid: " << cfg.global_nx << " x " << cfg.global_ny << "\n";
        std::cout << "  Patches: " << cfg.patches_x << " x " << cfg.patches_y << "\n";
        std::cout << "  Ranks: " << comm.size() << "\n";
        std::cout << "  Patches per rank: " << patches.size() << "\n";
    }

    // Use factory helpers to create stages
    auto pipe = parallel::pipeline(
        make_ghost_exchange_stage(cfg),
        compute_laplacian_t{},
        make_error_reduce_stage(cfg)
    );

    auto sched = parallel::scheduler_t{};
    auto profiler = perf::null_profiler_t{};
    parallel::execute(pipe, patches, comm, sched, profiler);

    double l2_error = patches[0].l2_error;
    double dx = cfg.domain_x / cfg.global_nx;

    if (comm.rank() == 0) {
        std::cout << "  L2 error: " << l2_error << "\n";
        double expected_error = dx * dx * std::pow(2.0 * M_PI, 4) / 12.0;
        if (l2_error < 2.0 * expected_error) {
            std::cout << "  PASSED (error within expected bounds)\n";
        } else {
            std::cout << "  FAILED (error larger than expected)\n";
        }
    }

    return 0;
}

// =============================================================================
// main_b: Minimal Lambda-Based Approach
// =============================================================================
// Reduces boilerplate by using lambdas for inline stage definitions.
// Good for: quick prototyping and minimal ceremony.

int main_b(int argc, char** argv) {
    auto mpi = mpi_context(argc, argv);
    auto comm = mpi.get_communicator();
    auto cfg = config_t{};
    auto patches = create_patches(cfg, comm.rank(), comm.size());
    double dx = cfg.domain_x / cfg.global_nx;
    double dy = cfg.domain_y / cfg.global_ny;

    if (comm.rank() == 0) {
        std::cout << "Laplacian 2D (Approach B: Minimal Lambda-Based)\n";
        std::cout << "  Grid: " << cfg.global_nx << " x " << cfg.global_ny << "\n";
        std::cout << "  Patches: " << cfg.patches_x << " x " << cfg.patches_y << "\n";
        std::cout << "  Ranks: " << comm.size() << "\n";
        std::cout << "  Patches per rank: " << patches.size() << "\n";
    }

    // Define stages inline with lambdas
    auto exchange = [num_ghosts=cfg.num_ghosts, nx=cfg.global_nx, ny=cfg.global_ny] {
        return ghost_exchange_t{.num_ghosts = num_ghosts, .global_nx = nx, .global_ny = ny};
    }();

    auto compute = compute_laplacian_t{};

    auto reduce = [dx, dy, ncells=cfg.global_nx * cfg.global_ny] {
        return error_reduce_t{.dx = dx, .dy = dy, .num_cells = ncells};
    }();

    auto pipe = parallel::pipeline(exchange, compute, reduce);
    auto sched = parallel::scheduler_t{};
    auto profiler = perf::null_profiler_t{};
    parallel::execute(pipe, patches, comm, sched, profiler);

    double l2_error = patches[0].l2_error;

    if (comm.rank() == 0) {
        std::cout << "  L2 error: " << l2_error << "\n";
        double expected_error = dx * dx * std::pow(2.0 * M_PI, 4) / 12.0;
        if (l2_error < 2.0 * expected_error) {
            std::cout << "  PASSED (error within expected bounds)\n";
        } else {
            std::cout << "  FAILED (error larger than expected)\n";
        }
    }

    return 0;
}

// =============================================================================
// main_c: Fluent DSL-Like Builder Approach
// =============================================================================
// Reduces boilerplate with a fluent builder interface.
// Good for: maximum convenience and readability.

int main_c(int argc, char** argv) {
    auto mpi = mpi_context(argc, argv);
    auto comm = mpi.get_communicator();

    auto builder = simulation_builder<patch_t>(comm.rank(), comm.size(), comm);
    builder
        .config(config_t{})
        .setup()
        .print_info("Laplacian 2D (Approach C: Fluent DSL Builder)");

    double l2_error = builder.run_exchange_compute_reduce();
    builder.validate_error(l2_error);

    return 0;
}

// =============================================================================
// main_d: General DSL Approach (Product-Based)
// =============================================================================
// Demonstrates the general simulation DSL with:
// - Cartesian decomposition: splits domain into 2x2 = 4 logical patches
// - Stage registration: exchange, compute, reduce
// - Product definition: named outputs for driver framework
// - Results extraction: std::any-based polymorphic returns
//
// DESIGN GOALS:
// 1. Works with any PatchType (1D/2D/3D patches, AMR, etc.)
// 2. Flexible decomposition strategies (cartesian, custom, AMR)
// 3. Named outputs for driver integration (avoid global state)
// 4. Clear separation: decomposition, stages, extraction
//
// USAGE PATTERNS:
//
//   // Simple 2D case
//   auto sim = simulation<patch_2d_t>(rank, size, comm)
//       .decompose_cartesian(uvec(4, 4), uvec(256, 256));  // 4x4 patches
//
//   // AMR with custom decomposition
//   auto sim = simulation<patch_amr_t>(rank, size, comm)
//       .decompose_custom([](int rank, int size) {
//           // Custom AMR decomposition returning vector<patch_amr_t>
//       });
//
//   // Define products for driver (e.g., "field", "error", "dt")
//   sim.define_product("solution", [](const auto& patches) {
//       return patches[0].u;  // Export solution array
//   });
//
//   // Execute stages and get results
//   auto results = sim.run();
//   double l2_err = std::any_cast<double>(results["l2_error"]);

int main_d(int argc, char** argv) {
    auto mpi = mpi_context(argc, argv);
    auto comm = mpi.get_communicator();

    // Create and configure simulation
    auto sim = simulation<patch_t>(comm.rank(), comm.size(), comm);
    sim.decompose_cartesian(uvec(2, 2), uvec(64, 64));

    // Define stages
    auto exchange = ghost_exchange_t{
        .num_ghosts = 1,
        .global_nx = 64,
        .global_ny = 64
    };
    auto compute = compute_laplacian_t{};
    auto reduce = error_reduce_t{
        .dx = 1.0 / 64,
        .dy = 1.0 / 64,
        .num_cells = 64 * 64
    };

    // Execute computation
    sim.execute(exchange, compute, reduce);

    // Define and extract products
    auto results = sim
        .define_product("l2_error", [](const std::vector<patch_t>& patches) -> std::any {
            return patches[0].l2_error;
        })
        .define_product("num_patches", [](const std::vector<patch_t>& patches) -> std::any {
            return patches.size();
        })
        .run();

    // Display results
    if (comm.rank() == 0) {
        std::cout << "Laplacian 2D (Approach D: General DSL with Products)\n";
        std::cout << "  Grid: 64 x 64\n";
        std::cout << "  Patches: 2 x 2\n";
        std::cout << "  Ranks: " << comm.size() << "\n";
        std::cout << "  Patches per rank: " << std::any_cast<size_t>(results["num_patches"]) << "\n";

        double l2_error = std::any_cast<double>(results["l2_error"]);
        std::cout << "  L2 error: " << l2_error << "\n";

        double dx = 1.0 / 64;
        double expected_error = dx * dx * std::pow(2.0 * M_PI, 4) / 12.0;
        if (l2_error < 2.0 * expected_error) {
            std::cout << "  PASSED (error within expected bounds)\n";
        } else {
            std::cout << "  FAILED (error larger than expected)\n";
        }
    }

    return 0;
}
