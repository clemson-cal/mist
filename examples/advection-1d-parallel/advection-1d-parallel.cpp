#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include "mist/core.hpp"
#include "mist/parallel.hpp"
#include "mist/parallel/algorithm.hpp"

using namespace mist;
using namespace mist::parallel;

// =============================================================================
// 1D Linear Advection - Algorithm API
// =============================================================================

struct ghost_message_t {
    int source_rank;
    std::vector<double> cells;
};

struct global_config_t {
    unsigned int total_zones = 10000;
    double domain_length = 1.0;
    double advection_velocity = 1.0;
    unsigned int ghost_depth = 2;
};

struct subdomain_state_t {
    std::vector<double> interior;
    double time = 0.0;
};

// =============================================================================
// Create advection algorithm using two_stage
// =============================================================================

auto make_advection_algorithm(const global_config_t& cfg, int nranks) {
    return two_stage(
        // Exchange phase: send boundaries to neighbors
        [&cfg, nranks](recording_communicator_t<ghost_message_t>& comm, int key, const subdomain_state_t& s) {
            // Send left boundary
            if (key > 0) {
                std::vector<double> left_boundary(cfg.ghost_depth);
                for (unsigned int i = 0; i < cfg.ghost_depth; ++i) {
                    left_boundary[i] = s.interior[i];
                }
                comm.send(key - 1, std::move(left_boundary));
                comm.recv(key - 1);
            }

            // Send right boundary
            if (key < nranks - 1) {
                std::vector<double> right_boundary(cfg.ghost_depth);
                for (unsigned int i = 0; i < cfg.ghost_depth; ++i) {
                    right_boundary[i] = s.interior[s.interior.size() - cfg.ghost_depth + i];
                }
                comm.send(key + 1, std::move(right_boundary));
                comm.recv(key + 1);
            }
        },
        // Compute phase: receive ghosts and compute timestep
        [&cfg](subdomain_state_t s, std::vector<ghost_message_t> messages) -> subdomain_state_t {
            // Extract received ghost data
            // Messages arrive in recv() order: left neighbor first, then right neighbor
            std::vector<double> left_ghost(cfg.ghost_depth, 0.0);
            std::vector<double> right_ghost(cfg.ghost_depth, 0.0);

            for (std::size_t i = 0; i < messages.size(); ++i) {
                auto& msg = messages[i];
                // Determine if this is left or right based on source_rank
                // Left neighbor has lower rank, right neighbor has higher rank
                // We can infer our rank from the message pattern
                if (i == 0 && !messages.empty()) {
                    // First message is from left neighbor (key-1)
                    left_ghost = std::move(msg.cells);
                } else {
                    // Second message (if any) is from right neighbor (key+1)
                    right_ghost = std::move(msg.cells);
                }
            }

            // Compute timestep
            const std::size_t L = left_ghost.size();
            const std::size_t I = s.interior.size();

            auto get_val = [&](std::size_t idx) -> double {
                if (idx < L) return left_ghost[idx];
                idx -= L;
                if (idx < I) return s.interior[idx];
                idx -= I;
                return right_ghost[idx];
            };

            double dx = cfg.domain_length / cfg.total_zones;
            double v = cfg.advection_velocity;
            double dt = 0.4 * dx / std::abs(v);

            std::vector<double> new_interior(I);
            for (std::size_t i = 0; i < I; ++i) {
                std::size_t idx = cfg.ghost_depth + i;

                if (v > 0) {
                    double flux_left = v * get_val(idx - 1);
                    double flux_right = v * get_val(idx);
                    new_interior[i] = get_val(idx) - dt / dx * (flux_right - flux_left);
                } else {
                    double flux_left = v * get_val(idx);
                    double flux_right = v * get_val(idx + 1);
                    new_interior[i] = get_val(idx) - dt / dx * (flux_right - flux_left);
                }
            }

            subdomain_state_t new_state;
            new_state.interior = std::move(new_interior);
            new_state.time = s.time + dt;

            return new_state;
        }
    );
}

// =============================================================================
// Domain decomposition
// =============================================================================

std::vector<subdomain_state_t> decompose_domain(const global_config_t& cfg, int nranks) {
    std::vector<subdomain_state_t> states;
    states.reserve(nranks);

    unsigned int zones_per_rank = cfg.total_zones / nranks;
    double dx = cfg.domain_length / cfg.total_zones;

    for (int rank = 0; rank < nranks; ++rank) {
        std::vector<double> interior(zones_per_rank);
        unsigned int start_zone = rank * zones_per_rank;

        for (unsigned int i = 0; i < zones_per_rank; ++i) {
            unsigned int global_i = start_zone + i;
            double x = (global_i + 0.5) * dx;
            interior[i] = std::sin(2.0 * M_PI * x / cfg.domain_length);
        }

        subdomain_state_t state;
        state.interior = std::move(interior);
        state.time = 0.0;

        states.push_back(std::move(state));
    }

    return states;
}

// =============================================================================
// Main driver
// =============================================================================

int main(int argc, char** argv) {
    bool use_parallel = true;
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--serial" || mode == "serial") {
            use_parallel = false;
        }
    }

    std::cout << "=== 1D Parallel Advection (Algorithm API) ===\n\n";

    global_config_t cfg;
    cfg.total_zones = 10000000;
    cfg.domain_length = 1.0;
    cfg.advection_velocity = 1.0;
    cfg.ghost_depth = 2;

    int nranks = 100;

    std::cout << "Configuration:\n";
    std::cout << "  Total zones: " << cfg.total_zones << "\n";
    std::cout << "  Subdomains: " << nranks << "\n";
    std::cout << "  Zones per subdomain: " << cfg.total_zones / nranks << "\n";
    std::cout << "  Ghost depth: " << cfg.ghost_depth << "\n\n";

    auto states = decompose_domain(cfg, nranks);

    // Create reusable algorithm
    auto algo = make_advection_algorithm(cfg, nranks);

    std::size_t nthreads = std::thread::hardware_concurrency() * 2;
    thread_pool_t pool(nthreads);
    sequential_scheduler_t seq;

    if (use_parallel) {
        std::cout << "Using " << nthreads << " threads (parallel)\n\n";
    } else {
        std::cout << "Running in serial mode (sequential scheduler)\n\n";
    }

    int nsteps = 100;
    double sum_mzps = 0.0;
    double min_mzps = std::numeric_limits<double>::infinity();
    double max_mzps = 0.0;
    int mzps_count = 0;

    for (int step = 0; step < nsteps; ++step) {
        std::cout << "Step " << step << " starting...\n" << std::flush;

        auto t0 = std::chrono::steady_clock::now();
        states = use_parallel
            ? execute(algo, pool, std::move(states))
            : execute(algo, seq, std::move(states));
        auto t1 = std::chrono::steady_clock::now();

        std::chrono::duration<double> elapsed = t1 - t0;
        double secs = elapsed.count();
        std::uint64_t zones_updated = cfg.total_zones;
        double mzps = secs > 0.0 ? (static_cast<double>(zones_updated) / (secs * 1e6)) : 0.0;

        sum_mzps += mzps;
        if (mzps < min_mzps) min_mzps = mzps;
        if (mzps > max_mzps) max_mzps = mzps;
        mzps_count++;

        std::cout << "Step " << step << " complete "
                  << "(elapsed = " << std::fixed << std::setprecision(6) << secs << " s, "
                  << "Mzps = " << std::setprecision(3) << mzps << ")\n" << std::flush;
    }

    std::cout << "\n=== Simulation Complete ===\n";
    if (mzps_count > 0) {
        std::cout << "Mzps summary: count=" << mzps_count
                  << " avg=" << std::fixed << std::setprecision(3) << (sum_mzps / mzps_count)
                  << " min=" << min_mzps
                  << " max=" << max_mzps << "\n";
    }
    std::cout << "Final time: " << states[0].time << "\n";

    return 0;
}
