// 2D Laplacian with distributed domain decomposition
// Demonstrates exchange-compute pipeline without the driver framework

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
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
// Main
// =============================================================================

int main(int argc, char** argv) {
#ifdef MIST_WITH_MPI
    MPI_Init(&argc, &argv);
    auto comm = comm_t::from_mpi(MPI_COMM_WORLD);
#else
    (void)argc;
    (void)argv;
    auto comm = comm_t{};
#endif

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

#ifdef MIST_WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
