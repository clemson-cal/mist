// =============================================================================
// 2D Laplacian with Distributed Domain Decomposition
// =============================================================================
//
// This example demonstrates a fluent DSL for defining parallel computation
// pipelines with minimal boilerplate. The computation pipeline consists of:
//
// 1. Grid Decomposition: Split a 64x64 domain into px×py patches, each
//    owned by a rank in the MPI communicator. The patch layout can be
//    specified at runtime via --patches=px,py (default: 2x2)
//
// 2. Ghost Exchange: Share boundary data between neighboring patches before
//    computation. Each patch requests ghost cells from its neighbors using
//    the exchange() stage.
//
// 3. Laplacian Computation: Apply a 5-point stencil to compute the discrete
//    Laplacian of u(x,y) = sin(2πx) sin(2πy). Results are compared against
//    the exact Laplacian for error analysis.
//
// 4. L2 Error Reduction: Combine local error contributions across all ranks
//    using a global reduce operation. The result is the RMS error of the
//    computed Laplacian.
//
// The fluent pipeline syntax allows composing these stages naturally:
//
//   transformation<patch_t>()
//       .exchange(ghost_exchange_t{...})
//       .compute(compute_laplacian_t{})
//       .reduce(error_reduce_t{...})
//       .execute(patches, comm, scheduler, profiler)
//
// Usage:
//   ./laplacian2d                 # 2x2 patches (default)
//   ./laplacian2d --patches=4,8   # 4x8 patches

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include "helpers.hpp"

using namespace mist;

// =============================================================================
// Configuration
// =============================================================================

struct config_t {
    int nx = 64, ny = 64;
    int px = 2, py = 2;
    int ng = 1;
    double lx = 1.0, ly = 1.0;
};

// =============================================================================
// Patch: owns a portion of the domain
// =============================================================================

struct patch_t {
    index_space_t<2> interior;
    array_t<double, 2> u;
    array_t<double, 2> lap;
    double dx;
    double dy;
    double l2_error = 0.0;
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
// Patch creation from index space
// =============================================================================

auto create_patch(const config_t& cfg, const index_space_t<2>& space) -> patch_t {
    double dx = cfg.lx / cfg.nx;
    double dy = cfg.ly / cfg.ny;
    auto with_ghosts = expand(space, cfg.ng);

    auto patch = patch_t{};
    patch.interior = space;
    patch.u = array_t<double, 2>(with_ghosts);
    patch.lap = array_t<double, 2>(space);
    patch.dx = dx;
    patch.dy = dy;

    for (auto idx : with_ghosts) {
        double x = (idx[0] + 0.5) * dx;
        double y = (idx[1] + 0.5) * dy;
        patch.u(idx) = initial_condition(x, y);
    }

    return patch;
}

// =============================================================================
// Exchange stage: share ghost zones between patches
// =============================================================================

struct ghost_exchange_t {
    static constexpr const char* name = "ghost_exchange";
    using context_type = patch_t;
    using value_type = double;
    static constexpr std::size_t rank = 2;

    int ng, nx, ny;

    auto provides(const patch_t& p) const -> array_view_t<const double, 2> {
        return p.u[p.interior];
    }

    void need(patch_t& p, std::function<void(array_view_t<value_type, rank>)> request) const {
        auto lo = start(p.interior);
        auto hi = upper(p.interior);

        if (lo[0] > 0)   request(p.u[ghost(p.interior, region::lo, axis::i, ng)]);
        if (hi[0] < nx)  request(p.u[ghost(p.interior, region::hi, axis::i, ng)]);
        if (lo[1] > 0)   request(p.u[ghost(p.interior, region::lo, axis::j, ng)]);
        if (hi[1] < ny)  request(p.u[ghost(p.interior, region::hi, axis::j, ng)]);
    }
};

// =============================================================================
// Compute stage: compute 5-point Laplacian stencil
// =============================================================================

struct compute_laplacian_t {
    static constexpr const char* name = "compute_laplacian";
    using context_type = patch_t;

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
    using context_type = patch_t;
    using value_type = double;

    double dx, dy;
    int nc;

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
        p.l2_error = std::sqrt(global_sum_sq / nc);
    }
};

// =============================================================================
// Output Helpers
// =============================================================================

auto print_patch_distribution(const comm_t& comm, const std::vector<patch_t>& patches) -> void {
    std::cout << "Rank " << comm.rank() << " owns " << patches.size() << " patch(es):\n";
    for (int i = 0; i < patches.size(); ++i) {
        auto lo = start(patches[i].interior);
        auto hi = upper(patches[i].interior);
        std::cout << "  Patch " << i << ": [" << lo[0] << "," << lo[1] << ") to ["
                  << hi[0] << "," << hi[1] << ")\n";
    }
}

auto print_results(const comm_t& comm, const config_t& cfg,
                   const std::vector<patch_t>& patches, double dx) -> void {
    if (comm.rank() != 0) return;

    std::cout << "\nLaplacian 2D example\n";
    std::cout << "  Grid: " << cfg.nx << " x " << cfg.ny << "\n";
    std::cout << "  Patches: " << cfg.px << " x " << cfg.py << "\n";
    std::cout << "  Ranks: " << comm.size() << "\n";
    std::cout << "  Patches per rank: " << patches.size() << "\n";
    std::cout << "  L2 error: " << patches[0].l2_error << "\n";

    double expected_error = dx * dx * std::pow(2.0 * M_PI, 4) / 12.0;
    auto status = patches[0].l2_error < 2.0 * expected_error ? "PASSED" : "FAILED";
    std::cout << "  " << status << " (error within expected bounds)\n";
}

// =============================================================================
// Command-Line Parsing
// =============================================================================

auto parse_patches(int argc, char** argv) -> std::pair<int, int> {
    int px = 2, py = 2;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 10) == "--patches=") {
            auto spec = arg.substr(10);
            size_t comma = spec.find(',');
            if (comma != std::string::npos) {
                try {
                    px = std::stoi(spec.substr(0, comma));
                    py = std::stoi(spec.substr(comma + 1));
                } catch (...) {
                    std::cerr << "Invalid --patches argument: " << arg << "\n";
                    std::cerr << "Expected format: --patches=px,py (e.g., --patches=4,6)\n";
                    std::exit(1);
                }
            } else {
                std::cerr << "Invalid --patches argument: " << arg << "\n";
                std::cerr << "Expected format: --patches=px,py (e.g., --patches=4,6)\n";
                std::exit(1);
            }
        }
    }
    return {px, py};
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char** argv) {
    auto mpi = mpi_context(argc, argv);
    auto comm = mpi.get_communicator();
    auto cfg = config_t{};
    auto [px, py] = parse_patches(argc, argv);
    cfg.px = px;
    cfg.py = py;

    // Decompose domain and create patches
    auto gs = index_space(ivec(0, 0), uvec(cfg.nx, cfg.ny));
    auto patches = grid(gs)
        .decompose(uvec(cfg.px, cfg.py))
        .distribute(comm)
        .map([&cfg](const auto& space) { return create_patch(cfg, space); });

    // Print patch distribution
    print_patch_distribution(comm, patches);

    // Define computation pipeline
    double dx = cfg.lx / cfg.nx;
    double dy = cfg.ly / cfg.ny;
    int nc = cfg.nx * cfg.ny;

    auto calc = transformation<patch_t>()
        .exchange(ghost_exchange_t{cfg.ng, cfg.nx, cfg.ny})
        .compute(compute_laplacian_t{})
        .reduce(error_reduce_t{dx, dy, nc});

    // Execute pipeline
    auto s = parallel::scheduler_t{};
    auto p = perf::null_profiler_t{};
    calc.execute(patches, comm, s, p);

    // Display results
    print_results(comm, cfg, patches, dx);

    return 0;
}
