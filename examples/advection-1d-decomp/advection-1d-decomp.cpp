#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include "mist/ascii_reader.hpp"
#include "mist/ascii_writer.hpp"
#include "mist/core.hpp"
#include "mist/driver.hpp"
#include "mist/ndarray.hpp"
#include "mist/parallel/pipeline.hpp"
#include "mist/serialize.hpp"

using namespace mist;

template<std::ranges::range R>
auto to_vector(R&& r) {
    auto v = std::vector<std::ranges::range_value_t<R>>{};
    for (auto&& e : r) {
        v.push_back(std::forward<decltype(e)>(e));
    }
    return v;
}

// =============================================================================
// Helper functions
// =============================================================================

auto cell_center_x(int i, double dx) -> double {
    return (i + 0.5) * dx;
}


// =============================================================================
// Patch state - unified context that flows through pipeline
// =============================================================================

struct patch_state_t {
    int rank = 0;
    int npartitions = 1;
    int num_zones = 0;
    int i0 = 0, i1 = 0;

    double v = 0.0;
    double dx = 0.0;
    double dt = 0.0;

    cached_t<double, 1> conserved;

    // Guard buffers with global index spaces (set at construction)
    cached_t<double, 1> l_recv;
    cached_t<double, 1> r_recv;

    patch_state_t() = default;

    patch_state_t(int rank_, int npartitions_, int num_zones_, cached_t<double, 1> conserved_)
        : rank(rank_)
        , npartitions(npartitions_)
        , num_zones(num_zones_)
        , conserved(std::move(conserved_))
    {
        auto s = space(conserved);
        i0 = start(s)[0];
        i1 = i0 + shape(s)[0];

        // Guard zones in global coordinates (with periodic wrapping)
        auto l_guard_start = (i0 - 1 + num_zones_) % num_zones_;
        auto r_guard_start = i1 % num_zones_;
        l_recv = cache(fill(index_space(ivec(l_guard_start), uvec(1)), 0.0), memory::host, exec::cpu);
        r_recv = cache(fill(index_space(ivec(r_guard_start), uvec(1)), 0.0), memory::host, exec::cpu);
    }

    void new_step(double v_, double dx_, double dt_) {
        v = v_;
        dx = dx_;
        dt = dt_;
    }
};

// =============================================================================
// Stage 1: Ghost exchange (Message stage)
// =============================================================================

struct ghost_exchange_t {
    static void need(patch_state_t& ctx, auto request) {
        request(ctx.l_recv);
        request(ctx.r_recv);
    }

    static void fill(patch_state_t& ctx, auto provide) {
        provide(ctx.conserved);
    }
};

// =============================================================================
// Stage 2: Flux and update (Compute stage)
// =============================================================================

struct flux_and_update_t {
    static auto value(patch_state_t ctx) -> patch_state_t {
        if (ctx.v > 0) {
            for (auto i = ctx.i1 - 1; i >= ctx.i0; --i) {
                auto u_l = (i == ctx.i0) ? ctx.l_recv[0] : ctx.conserved[i - 1];
                auto flux_l = ctx.v * u_l;
                auto flux_r = ctx.v * ctx.conserved[i];
                ctx.conserved[i] = ctx.conserved[i] - ctx.dt / ctx.dx * (flux_r - flux_l);
            }
        } else {
            for (auto i = ctx.i0; i < ctx.i1; ++i) {
                auto u_r = (i == ctx.i1 - 1) ? ctx.r_recv[0] : ctx.conserved[i + 1];
                auto flux_l = ctx.v * ctx.conserved[i];
                auto flux_r = ctx.v * u_r;
                ctx.conserved[i] = ctx.conserved[i] - ctx.dt / ctx.dx * (flux_r - flux_l);
            }
        }
        return ctx;
    }
};

// =============================================================================
// Custom serialization for patch_state_t
// =============================================================================

template<ArchiveWriter A>
void serialize(A& ar, const patch_state_t& patch) {
    ar.begin_group();
    serialize(ar, "rank", patch.rank);
    serialize(ar, "npartitions", patch.npartitions);
    serialize(ar, "num_zones", patch.num_zones);
    serialize(ar, "conserved", patch.conserved);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_state_t& patch) -> bool {
    if (!ar.begin_group()) return false;
    auto rank = 0;
    auto npartitions = 1;
    auto num_zones = 0;
    auto conserved = cached_t<double, 1>{};
    deserialize(ar, "rank", rank);
    deserialize(ar, "npartitions", npartitions);
    deserialize(ar, "num_zones", num_zones);
    deserialize(ar, "conserved", conserved);
    ar.end_group();
    patch = patch_state_t(rank, npartitions, num_zones, std::move(conserved));
    return true;
}

// =============================================================================
// 1D Linear Advection Physics Module with Domain Decomposition
// =============================================================================

struct advection {

    struct config_t {
        int rk_order = 1;
        double cfl = 0.4;
        double advection_velocity = 1.0;

        auto fields() const {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("advection_velocity", advection_velocity)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("advection_velocity", advection_velocity)
            );
        }
    };

    struct initial_t {
        unsigned int num_zones = 200;
        unsigned int num_partitions = 4;
        double domain_length = 1.0;

        auto fields() const {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("num_partitions", num_partitions),
                field("domain_length", domain_length)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("num_zones", num_zones),
                field("num_partitions", num_partitions),
                field("domain_length", domain_length)
            );
        }
    };

    struct state_t {
        std::vector<patch_state_t> patches;
        double time;

        auto fields() const {
            return std::make_tuple(
                field("patches", patches),
                field("time", time)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("patches", patches),
                field("time", time)
            );
        }
    };

    using product_t = std::vector<cached_t<double, 1>>;

    struct exec_context_t {
        const config_t& config;
        const initial_t& initial;
        mutable parallel::scheduler_t scheduler;

        exec_context_t(const config_t& cfg, const initial_t& ini)
            : config(cfg), initial(ini) {}

        void set_num_threads(std::size_t n) {
            scheduler.set_num_threads(n);
        }
    };
};

// =============================================================================
// Physics interface implementation
// =============================================================================

auto default_physics_config(std::type_identity<advection>) -> advection::config_t {
    return {.rk_order = 1, .cfl = 0.4, .advection_velocity = 1.0};
}

auto default_initial_config(std::type_identity<advection>) -> advection::initial_t {
    return {.num_zones = 200, .num_partitions = 4, .domain_length = 1.0};
}

auto initial_state(const advection::exec_context_t& ctx) -> advection::state_t {
    using std::views::iota;
    using std::views::transform;

    auto& ini = ctx.initial;
    auto npartitions = static_cast<int>(ini.num_partitions);
    auto num_zones = static_cast<int>(ini.num_zones);
    auto global_space = index_space(ivec(0), uvec(ini.num_zones));
    auto dx = ini.domain_length / ini.num_zones;
    auto L = ini.domain_length;

    auto patches = to_vector(iota(0, npartitions) | transform([&](int p) {
        auto s = subspace(global_space, npartitions, p, 0);
        auto u = lazy(s, [&](auto i) {
            return std::sin(2.0 * M_PI * cell_center_x(i[0], dx) / L);
        });
        return patch_state_t(p, npartitions, num_zones, cache(u, memory::host, exec::cpu));
    }));
    return {std::move(patches), 0.0};
}

// Request structure for ghost exchange
struct ghost_request_t {
    int requester;
    cached_t<double, 1>* buffer;
    index_space_t<1> requested_space;
};

// Unigrid Cartesian topology for 1D periodic domain
struct unigrid_topology_1d {
    int num_zones;

    // Find which patch owns a given global index
    auto owner(int global_index, const std::vector<patch_state_t>& patches) const -> int {
        for (std::size_t p = 0; p < patches.size(); ++p) {
            if (global_index >= patches[p].i0 && global_index < patches[p].i1) {
                return static_cast<int>(p);
            }
        }
        return -1; // Should not happen with valid periodic indices
    }

    // Execute ghost exchange: collect requests, route to owners, fill
    void exchange(std::vector<patch_state_t>& patches) const {
        auto requests = std::vector<ghost_request_t>{};

        for (std::size_t p = 0; p < patches.size(); ++p) {
            ghost_exchange_t::need(patches[p], [&](cached_t<double, 1>& buf) {
                requests.push_back({static_cast<int>(p), &buf, space(buf)});
            });
        }

        // Route each request to owning patch(es) and fill
        for (auto& req : requests) {
            auto req_start = start(req.requested_space)[0];
            auto req_size = shape(req.requested_space)[0];

            for (std::size_t i = 0; i < req_size; ++i) {
                auto global_idx = (req_start + static_cast<int>(i) + num_zones) % num_zones;
                auto owner_idx = owner(global_idx, patches);

                ghost_exchange_t::fill(patches[owner_idx], [&](cached_t<double, 1>& src) {
                    auto src_space = space(src);
                    if (global_idx >= start(src_space)[0] && global_idx < start(src_space)[0] + static_cast<int>(shape(src_space)[0])) {
                        (*req.buffer)[req_start + static_cast<int>(i)] = src[global_idx];
                    }
                });
            }
        }
    }
};

void advance(
    advection::state_t& state,
    double dt,
    const advection::exec_context_t& exec
) {
    if (exec.config.rk_order != 1) {
        throw std::runtime_error("only rk_order=1 (forward Euler) is supported");
    }

    auto dx = exec.initial.domain_length / exec.initial.num_zones;
    auto v = exec.config.advection_velocity;

    // Set step parameters on all patches
    for (auto& patch : state.patches) {
        patch.new_step(v, dx, dt);
    }

    // Ghost exchange via topology
    auto topology = unigrid_topology_1d{static_cast<int>(exec.initial.num_zones)};
    topology.exchange(state.patches);

    // Compute flux and update
    for (auto& patch : state.patches) {
        patch = flux_and_update_t::value(std::move(patch));
    }

    state.time += dt;
}

auto courant_time(
    const advection::state_t& state,
    const advection::exec_context_t& ctx
) -> double {
    const auto dx = ctx.initial.domain_length / ctx.initial.num_zones;
    const auto v = std::abs(ctx.config.advection_velocity);
    return ctx.config.cfl * dx / v;
}

auto zone_count(const advection::state_t& state, const advection::exec_context_t& ctx) -> std::size_t {
    return ctx.initial.num_zones;
}

auto names_of_time(std::type_identity<advection>) -> std::vector<std::string> {
    return {"t"};
}

auto names_of_timeseries(std::type_identity<advection>) -> std::vector<std::string> {
    return {"time", "total_mass", "min_value", "max_value"};
}

auto names_of_products(std::type_identity<advection>) -> std::vector<std::string> {
    return {"concentration", "cell_x", "advection_velocity"};
}

auto get_time(
    const advection::state_t& state,
    const std::string& name
) -> double {
    if (name == "t") {
        return state.time;
    }
    throw std::runtime_error("unknown time variable: " + name);
}

auto get_timeseries(
    const advection::config_t& cfg,
    const advection::initial_t& ini,
    const advection::state_t& state,
    const std::string& name
) -> double {
    using std::views::transform;

    if (name == "time") {
        return state.time;
    }

    const auto dx = ini.domain_length / ini.num_zones;
    const auto& patches = state.patches;
    auto sums = patches | transform([](const auto& p) { return sum(p.conserved); });
    auto mins = patches | transform([](const auto& p) { return min(p.conserved); });
    auto maxs = patches | transform([](const auto& p) { return max(p.conserved); });

    if (name == "total_mass") {
        return std::accumulate(sums.begin(), sums.end(), 0.0) * dx;
    } else if (name == "min_value") {
        return std::ranges::min(mins);
    } else if (name == "max_value") {
        return std::ranges::max(maxs);
    }

    throw std::runtime_error("unknown timeseries column: " + name);
}

auto get_product(
    const advection::state_t& state,
    const std::string& name,
    const advection::exec_context_t& ctx
) -> advection::product_t {
    using std::views::transform;

    auto dx = ctx.initial.domain_length / ctx.initial.num_zones;
    auto v = ctx.config.advection_velocity;

    if (name == "concentration") {
        return to_vector(state.patches | transform([](const auto& p) {
            return cache(map(p.conserved, std::identity{}), memory::host, exec::cpu);
        }));
    } else if (name == "cell_x") {
        return to_vector(state.patches | transform([dx](const auto& p) {
            auto s = space(p.conserved);
            return cache(lazy(s, [dx](auto idx) { return cell_center_x(idx[0], dx); }), memory::host, exec::cpu);
        }));
    } else if (name == "advection_velocity") {
        return to_vector(state.patches | transform([v](const auto& p) {
            auto s = space(p.conserved);
            return cache(fill(s, v), memory::host, exec::cpu);
        }));
    }
    throw std::runtime_error("unknown product: " + name);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
    // Create program with default physics and initial configs
    mist::program_t<advection> prog;
    prog.physics = default_physics_config(std::type_identity<advection>{});
    prog.initial = default_initial_config(std::type_identity<advection>{});

    // Run interactive simulation
    auto final_state = mist::run(prog);

    return 0;
}
