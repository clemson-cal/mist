#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
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
// Patch - unified context that flows through pipeline
// =============================================================================

struct patch_t {
    index_space_t<1> interior;

    double v = 0.0;
    double dx = 0.0;
    double dt = 0.0;

    cached_t<double, 1> conserved;
    cached_t<double, 1> godunov_flux;
    cached_t<double, 1> l_recv;
    cached_t<double, 1> r_recv;

    patch_t() = default;

    patch_t(index_space_t<1> s)
        : interior(s)
        , conserved(cache(zeros<double>(s), memory::host, exec::cpu))
        , godunov_flux(cache(zeros<double>(index_space(start(s), uvec(upper(s)[0] - start(s)[0] + 1))), memory::host, exec::cpu))
        , l_recv(cache(zeros<double>(index_space(start(s) - ivec(1), uvec(1))), memory::host, exec::cpu))
        , r_recv(cache(zeros<double>(index_space(upper(s), uvec(1))), memory::host, exec::cpu))
    {
    }
};

// =============================================================================
// Stage 1: Ghost exchange (Message stage)
// =============================================================================

struct ghost_exchange_t {
    auto provides(const patch_t& p) const -> index_space_t<1> {
        return space(p.conserved);
    }

    void need(patch_t& p, auto request) const {
        request(p.l_recv);
        request(p.r_recv);
    }

    void fill(patch_t& p, auto provide) const {
        provide(p.conserved);
    }
};

// =============================================================================
// Stage 2: Flux and update (Compute stage)
// =============================================================================

struct compute_flux_t {
    auto value(patch_t p) const -> patch_t {
        auto lo = start(p.interior)[0];
        auto hi = upper(p.interior)[0];
        auto v = p.v;
        auto& fhat = p.godunov_flux;
        auto& u = p.conserved;
        auto& ul = p.l_recv;
        auto& ur = p.r_recv;

        if (v > 0) {
            fhat[lo] = v * ul[lo - 1];
            for_each(index_space(ivec(lo + 1), uvec(hi - lo)), [&](ivec_t<1> i) {
                fhat(i) = v * u(i - ivec(1));
            });
        } else {
            for_each(index_space(ivec(lo), uvec(hi - lo)), [&](ivec_t<1> i) {
                fhat(i) = v * u(i);
            });
            fhat[hi] = v * ur[hi];
        }

        return p;
    }
};

struct update_conserved_t {
    auto value(patch_t p) const -> patch_t {
        auto& u = p.conserved;
        auto& fhat = p.godunov_flux;
        auto dtdx = p.dt / p.dx;

        for_each(p.interior, [&](ivec_t<1> i) {
            u(i) -= dtdx * (fhat(i + ivec(1)) - fhat(i));
        });
        return p;
    }
};

// =============================================================================
// Stage 3: Compute local dt (Compute stage)
// =============================================================================

struct compute_local_dt_t {
    double cfl;
    double v;
    double dx;

    auto value(patch_t p) const -> patch_t {
        p.v = v;
        p.dx = dx;
        p.dt = cfl * dx / std::abs(v);
        return p;
    }
};

// =============================================================================
// Stage 4: Global dt reduction (Reduce stage)
// =============================================================================

struct global_dt_t {
    using value_type = double;

    static double init() { return std::numeric_limits<double>::max(); }

    double reduce(double acc, const patch_t& p) const {
        return std::min(acc, p.dt);
    }

    void finalize(double dt, patch_t& p) const {
        p.dt = dt;
    }
};

// =============================================================================
// Custom serialization for patch_t
// =============================================================================

template<ArchiveWriter A>
void serialize(A& ar, const patch_t& p) {
    ar.begin_group();
    serialize(ar, "conserved", p.conserved);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_t& p) -> bool {
    if (!ar.begin_group()) return false;
    auto u = cached_t<double, 1>{};
    deserialize(ar, "conserved", u);
    ar.end_group();
    p = patch_t(space(u));
    copy(p.conserved, u);
    return true;
}

// =============================================================================
// Unigrid Cartesian topology for 1D domain with outflow boundaries
// =============================================================================

struct unigrid_topology_1d {
    using buffer_t = cached_t<double, 1>;
    using space_t = index_space_t<1>;

    void copy(buffer_t& dst, const buffer_t& src, space_t) const {
        copy_overlapping(dst, src);
    }

    bool connected(space_t a, space_t b) const {
        return (upper(a)[0] == start(b)[0]) || (upper(b)[0] == start(a)[0]);
    }
};

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
        std::vector<patch_t> patches;
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
        unigrid_topology_1d topology;

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

struct initial_state_t {
    double dx;
    double L;

    auto value(patch_t p) const -> patch_t {
        auto& u = p.conserved;
        for_each(p.interior, [&](ivec_t<1> i) {
            u(i) = std::sin(2.0 * M_PI * cell_center_x(i[0], dx) / L);
        });
        return p;
    }
};

auto initial_state(const advection::exec_context_t& ctx) -> advection::state_t {
    using std::views::iota;
    using std::views::transform;

    auto& ini = ctx.initial;
    auto np = static_cast<int>(ini.num_partitions);
    auto S = index_space(ivec(0), uvec(ini.num_zones));
    auto dx = ini.domain_length / ini.num_zones;
    auto L = ini.domain_length;

    auto patches = to_vector(iota(0, np) | transform([&](int p) {
        return patch_t(subspace(S, np, p, 0));
    }));

    parallel::execute(initial_state_t{dx, L}, patches, ctx.topology, ctx.scheduler);

    return {std::move(patches), 0.0};
}

void advance(advection::state_t& state, const advection::exec_context_t& ctx) {
    if (ctx.config.rk_order != 1) {
        throw std::runtime_error("only rk_order=1 (forward Euler) is supported");
    }

    auto dx = ctx.initial.domain_length / ctx.initial.num_zones;
    auto v = ctx.config.advection_velocity;
    auto cfl = ctx.config.cfl;

    auto pipe = parallel::pipeline(
        compute_local_dt_t{cfl, v, dx},
        global_dt_t{},
        ghost_exchange_t{},
        compute_flux_t{},
        update_conserved_t{}
    );
    parallel::execute(pipe, state.patches, ctx.topology, ctx.scheduler);

    state.time += state.patches[0].dt;
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

    if (name == "time")
        return state.time;

    const auto dx = ini.domain_length / ini.num_zones;
    const auto& patches = state.patches;
    auto sums = patches | transform([](const auto& p) { return sum(p.conserved); });
    auto mins = patches | transform([](const auto& p) { return min(p.conserved); });
    auto maxs = patches | transform([](const auto& p) { return max(p.conserved); });

    if (name == "total_mass")
        return std::accumulate(sums.begin(), sums.end(), 0.0) * dx;
    if (name == "min_value")
        return std::ranges::min(mins);
    if (name == "max_value")
        return std::ranges::max(maxs);

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
    }
    if (name == "cell_x") {
        return to_vector(state.patches | transform([dx](const auto& p) {
            auto s = space(p.conserved);
            return cache(lazy(s, [dx](auto idx) { return cell_center_x(idx[0], dx); }), memory::host, exec::cpu);
        }));
    }
    if (name == "advection_velocity") {
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
