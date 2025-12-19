#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include "mist/core.hpp"
#include "mist/driver/physics_impl.hpp"
#include "mist/driver/repl_session.hpp"
#include "mist/driver/socket_session.hpp"
#include "mist/ndarray.hpp"
#include "mist/pipeline.hpp"
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

    cached_t<double, 1> cons;
    cached_t<double, 1> fhat;

    patch_t() = default;

    patch_t(index_space_t<1> s)
        : interior(s)
        , cons(cache(zeros<double>(expand(s, 1)), memory::host, exec::cpu))
        , fhat(cache(zeros<double>(nudge(s, ivec(0), ivec(1))), memory::host, exec::cpu))
    {
    }
};

// =============================================================================
// Pipeline stages
// =============================================================================

struct initial_state_t {
    static constexpr const char* name = "initial_state";
    double dx;
    double L;

    auto value(patch_t p) const -> patch_t {
        auto& u = p.cons;
        for_each(p.interior, [&](ivec_t<1> i) {
            u(i) = std::sin(2.0 * M_PI * cell_center_x(i[0], dx) / L);
        });
        return p;
    }
};

struct compute_local_dt_t {
    static constexpr const char* name = "compute_local_dt";
    double cfl;
    double v;
    double dx;
    double dt_max;

    auto value(patch_t p) const -> patch_t {
        p.v = v;
        p.dx = dx;
        p.dt = std::min(cfl * dx / std::abs(v), dt_max);
        return p;
    }
};

struct global_dt_t {
    static constexpr const char* name = "global_dt";
    using value_type = double;

    static double init() {
        return std::numeric_limits<double>::max();
    }

    double reduce(double acc, const patch_t& p) const {
        return std::min(acc, p.dt);
    }

    void finalize(double dt, patch_t& p) const {
        p.dt = dt;
    }
};

struct ghost_exchange_t {
    static constexpr const char* name = "ghost_exchange";
    using space_t = index_space_t<1>;
    using buffer_t = array_view_t<double, 1>;

    auto provides(const patch_t& p) const -> space_t {
        return p.interior;
    }

    void need(patch_t& p, auto request) const {
        auto lo = start(p.interior);
        auto hi = upper(p.interior);
        auto l_guard = index_space(lo - ivec(1), uvec(1));
        auto r_guard = index_space(hi - ivec(0), uvec(1));
        request(p.cons[l_guard]);
        request(p.cons[r_guard]);
    }

    auto data(const patch_t& p) const -> array_view_t<const double, 1> {
        return p.cons[p.interior];
    }
};

// Unfused stages (for comparison)
struct compute_flux_t {
    static constexpr const char* name = "compute_flux";
    auto value(patch_t p) const -> patch_t {
        auto v = p.v;
        auto& u = p.cons;
        auto& f = p.fhat;

        if (v > 0.0) {
            for_each(space(f), [&](ivec_t<1> i) {
                f(i) = v * u(i - ivec(1));
            });
        } else {
            for_each(space(f), [&](ivec_t<1> i) {
                f(i) = v * u(i);
            });
        }
        return p;
    }
};

struct update_conserved_t {
    static constexpr const char* name = "update_conserved";
    auto value(patch_t p) const -> patch_t {
        auto dtdx = p.dt / p.dx;
        auto& u = p.cons;
        auto& f = p.fhat;

        for_each(p.interior, [&](ivec_t<1> i) {
            u(i) -= dtdx * (f(i + ivec(1)) - f(i));
        });
        return p;
    }
};

// Fused flux + update stage (better cache utilization)
struct flux_and_update_t {
    static constexpr const char* name = "flux_and_update";
    auto value(patch_t p) const -> patch_t {
        auto v = p.v;
        auto dtdx = p.dt / p.dx;
        auto& u = p.cons;
        auto i0 = start(p.interior)[0];
        auto i1 = upper(p.interior)[0];

        if (v > 0.0) {
            // Backward iteration for upwind stability
            for (int i = i1 - 1; i >= i0; --i) {
                auto flux_l = v * u[i - 1];
                auto flux_r = v * u[i];
                u[i] = u[i] - dtdx * (flux_r - flux_l);
            }
        } else {
            // Forward iteration for upwind stability
            for (int i = i0; i < i1; ++i) {
                auto flux_l = v * u[i];
                auto flux_r = v * u[i + 1];
                u[i] = u[i] - dtdx * (flux_r - flux_l);
            }
        }
        return p;
    }
};

// =============================================================================
// Custom serialization for patch_t
// =============================================================================

template<ArchiveWriter A>
void serialize(A& ar, const patch_t& p) {
    ar.begin_group();
    auto interior = cached_t<double, 1>(p.interior, memory::host);
    copy(interior[p.interior], p.cons[p.interior]);
    serialize(ar, "cons", interior);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_t& p) -> bool {
    if (!ar.begin_group()) return false;
    auto interior = cached_t<double, 1>{};
    deserialize(ar, "cons", interior);
    ar.end_group();
    p = patch_t(space(interior));
    copy(p.cons[p.interior], interior);
    return true;
}

// =============================================================================
// 1D Linear Advection Physics Module with Domain Decomposition
// =============================================================================

struct advection {

    struct config_t {
        int rk_order = 1;
        double cfl = 0.4;
        double wavespeed = 1.0;
        bool use_flux_buffer = false;

        auto fields() const {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("wavespeed", wavespeed),
                field("use_flux_buffer", use_flux_buffer)
            );
        }

        auto fields() {
            return std::make_tuple(
                field("rk_order", rk_order),
                field("cfl", cfl),
                field("wavespeed", wavespeed),
                field("use_flux_buffer", use_flux_buffer)
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
        mutable perf::profiler_t profiler;

        exec_context_t(const config_t& cfg, const initial_t& ini)
            : config(cfg), initial(ini) {}

        void set_num_threads(std::size_t n) {
            scheduler.set_num_threads(n);
        }

        template<parallel::Pipeline P>
        void execute(P pipeline, std::vector<patch_t>& patches) const {
            parallel::execute(pipeline, patches, scheduler, profiler);
        }
    };
};

// =============================================================================
// Physics interface implementation
// =============================================================================

auto default_physics_config(std::type_identity<advection>) -> advection::config_t {
    return {.rk_order = 1, .cfl = 0.4, .wavespeed = 1.0};
}

auto default_initial_config(std::type_identity<advection>) -> advection::initial_t {
    return {.num_zones = 200, .num_partitions = 4, .domain_length = 1.0};
}

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

    parallel::execute(initial_state_t{dx, L}, patches, ctx.scheduler, ctx.profiler);

    return {std::move(patches), 0.0};
}

void advance(advection::state_t& state, const advection::exec_context_t& ctx, double dt_max) {
    if (ctx.config.rk_order != 1) {
        throw std::runtime_error("only rk_order=1 (forward Euler) is supported");
    }

    auto dx = ctx.initial.domain_length / ctx.initial.num_zones;
    auto v = ctx.config.wavespeed;
    auto cfl = ctx.config.cfl;

    if (ctx.config.use_flux_buffer) { // Use of separate flux buffer (poor scaling -- memory bandwidth?)
        auto pipeline = parallel::pipeline(
            ghost_exchange_t{},
            compute_local_dt_t{cfl, v, dx, dt_max},
            compute_flux_t{},
            update_conserved_t{}
        );
        ctx.execute(pipeline, state.patches);
    } else { // No use of separate flux buffer (good scaling)
        auto pipeline = parallel::pipeline(
            ghost_exchange_t{},
            compute_local_dt_t{cfl, v, dx, dt_max},
            flux_and_update_t{}
        );
        ctx.execute(pipeline, state.patches);
    }
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
    return {"concentration", "cell_x", "wavespeed"};
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
    const advection::state_t& state,
    const std::string& name,
    const advection::exec_context_t& ctx
) -> double {
    using std::views::transform;

    if (name == "time")
        return state.time;

    const auto dx = ctx.initial.domain_length / ctx.initial.num_zones;
    const auto& patches = state.patches;
    auto sums = patches | transform([](const auto& p) { return sum(p.cons); });
    auto mins = patches | transform([](const auto& p) { return min(p.cons); });
    auto maxs = patches | transform([](const auto& p) { return max(p.cons); });

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
    auto v = ctx.config.wavespeed;

    if (name == "concentration") {
        return to_vector(state.patches | transform([](const auto& p) {
            auto result = cached_t<double, 1>(p.interior, memory::host);
            copy(result[p.interior], p.cons[p.interior]);
            return result;
        }));
    }
    if (name == "cell_x") {
        return to_vector(state.patches | transform([dx](const auto& p) {
            return cache(lazy(p.interior, [dx](auto idx) { return cell_center_x(idx[0], dx); }), memory::host, exec::cpu);
        }));
    }
    if (name == "wavespeed") {
        return to_vector(state.patches | transform([v](const auto& p) {
            return cache(fill(p.interior, v), memory::host, exec::cpu);
        }));
    }
    throw std::runtime_error("unknown product: " + name);
}

auto get_profiler_data(const advection::exec_context_t& ctx)
    -> std::map<std::string, perf::profile_entry_t>
{
    return ctx.profiler.data();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[])
{
    auto use_socket = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--socket") == 0 || std::strcmp(argv[i], "-s") == 0) {
            use_socket = true;
        }
    }

    auto physics = mist::driver::make_physics<advection>();
    auto state = mist::driver::state_t{};
    auto engine = mist::driver::engine_t{state, *physics};

    if (use_socket) {
        auto session = mist::driver::socket_session_t{engine};
        return session.run();
    } else {
        auto session = mist::driver::repl_session_t{engine};
        return session.run();
    }
}
