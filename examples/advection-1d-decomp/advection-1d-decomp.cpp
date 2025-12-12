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
    int num_zones = 0;
    index_space_t<1> interior;

    double v = 0.0;
    double dx = 0.0;
    double dt = 0.0;

    cached_t<double, 1> conserved;
    cached_t<double, 1> l_recv;
    cached_t<double, 1> r_recv;

    patch_state_t() = default;

    patch_state_t(int num_zones_, cached_t<double, 1> conserved_)
        : num_zones(num_zones_)
        , interior(space(conserved_))
        , conserved(std::move(conserved_))
    {
        // Guard zones at virtual indices (may be outside [0, num_zones) for boundary patches)
        l_recv = cache(zeros<double>(index_space(start(interior) - ivec(1), uvec(1))), memory::host, exec::cpu);
        r_recv = cache(zeros<double>(index_space(upper(interior), uvec(1))), memory::host, exec::cpu);
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
    static auto provides(const patch_state_t& ctx) -> index_space_t<1> {
        return space(ctx.conserved);
    }

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
        auto i0 = start(ctx.interior)[0];
        auto i1 = upper(ctx.interior)[0];

        // Combine guards and interior into unified view
        auto u = union_<double, 1>(ctx.l_recv, ctx.conserved, ctx.r_recv);

        // Outflow BC: clamp indices to valid range
        auto get = [&](int i) {
            return u(ivec(std::clamp(i, 0, ctx.num_zones - 1)));
        };

        if (ctx.v > 0) {
            for (auto i = i1 - 1; i >= i0; --i) {
                auto flux_l = ctx.v * get(i - 1);
                auto flux_r = ctx.v * ctx.conserved[i];
                ctx.conserved[i] = ctx.conserved[i] - ctx.dt / ctx.dx * (flux_r - flux_l);
            }
        } else {
            for (auto i = i0; i < i1; ++i) {
                auto flux_l = ctx.v * ctx.conserved[i];
                auto flux_r = ctx.v * get(i + 1);
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
    serialize(ar, "num_zones", patch.num_zones);
    serialize(ar, "conserved", patch.conserved);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_state_t& patch) -> bool {
    if (!ar.begin_group()) return false;
    auto num_zones = 0;
    auto conserved = cached_t<double, 1>{};
    deserialize(ar, "num_zones", num_zones);
    deserialize(ar, "conserved", conserved);
    ar.end_group();
    patch = patch_state_t(num_zones, std::move(conserved));
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
        return patch_state_t(num_zones, cache(u, memory::host, exec::cpu));
    }));
    return {std::move(patches), 0.0};
}

// Unigrid Cartesian topology for 1D domain with outflow boundaries
// Satisfies parallel::Topology<unigrid_topology_1d, patch_state_t>
struct unigrid_topology_1d {
    using buffer_t = cached_t<double, 1>;
    using space_t = index_space_t<1>;

    // Copies overlapping data from source to destination
    void copy(buffer_t& dst, const buffer_t& src, space_t) const {
        copy_overlapping(dst, src);
    }

    // Returns true if two spaces are adjacent (could exchange guards)
    bool connected(space_t a, space_t b) const {
        return (upper(a)[0] == start(b)[0]) || (upper(b)[0] == start(a)[0]);
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

    // Execute pipeline: Exchange stage (ghost_exchange_t) + Compute stage (flux_and_update_t)
    using advection_pipeline = parallel::pipeline<ghost_exchange_t, flux_and_update_t>;
    auto topology = unigrid_topology_1d{};
    parallel::execute(advection_pipeline{}, state.patches, topology, exec.scheduler);

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
