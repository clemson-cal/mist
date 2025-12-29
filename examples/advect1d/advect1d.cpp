#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include "mist/comm.hpp"
#include "mist/core.hpp"
#include "mist/driver/physics_impl.hpp"
#include "mist/driver/repl_session.hpp"
#include "mist/driver/socket_session.hpp"
#include "mist/ndarray.hpp"
#include "mist/pipeline.hpp"
#include "mist/archive.hpp"

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
    double cfl = 0.0;
    double dx = 0.0;
    double dt = 0.0;
    double L = 0.0;

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
    using context_type = patch_t;

    auto value(patch_t p) const -> patch_t {
        auto& u = p.cons;
        auto dx = p.dx;
        auto L = p.L;
        for_each(p.interior, [&](ivec_t<1> i) {
            u(i) = std::sin(2.0 * M_PI * cell_center_x(i[0], dx) / L);
        });
        return p;
    }
};

struct compute_local_dt_t {
    static constexpr const char* name = "compute_local_dt";
    using context_type = patch_t;
    double dt_max;

    auto value(patch_t p) const -> patch_t {
        p.dt = std::min(p.cfl * p.dx / std::abs(p.v), dt_max);
        return p;
    }
};

struct global_dt_t {
    static constexpr const char* name = "global_dt";
    using context_type = patch_t;
    using value_type = double;

    static auto init() -> double {
        return std::numeric_limits<double>::max();
    }

    static auto combine(double a, double b) -> double {
        return std::min(a, b);
    }

    auto extract(const patch_t& p) const -> double {
        return p.dt;
    }

    void finalize(double dt, patch_t& p) const {
        p.dt = dt;
    }
};

struct ghost_exchange_t {
    static constexpr const char* name = "ghost_exchange";
    using context_type = patch_t;
    using value_type = double;
    static constexpr std::size_t rank = 1;

    auto provides(const patch_t& p) const -> array_view_t<const double, 1> {
        return p.cons[p.interior];
    }

    void need(patch_t& p, std::function<void(array_view_t<value_type, rank>)> request) const {
        auto lo = start(p.interior);
        auto hi = upper(p.interior);
        auto l_guard = index_space(lo - ivec(1), uvec(1));
        auto r_guard = index_space(hi - ivec(0), uvec(1));
        request(p.cons[l_guard]);
        request(p.cons[r_guard]);
    }
};

// Unfused stages (for comparison)
struct compute_flux_t {
    static constexpr const char* name = "compute_flux";
    using context_type = patch_t;
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
    using context_type = patch_t;
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
    using context_type = patch_t;
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
// Truthful implementation for patch_t
// =============================================================================

// The serializable "truth" of a patch (excludes derived fields like fhat)
struct patch_truth_t {
    cached_t<double, 1> cons;  // interior conserved data only
    double v = 0.0;
    double cfl = 0.0;
    double dx = 0.0;
    double L = 0.0;
};

inline auto fields(patch_truth_t& t) {
    return std::make_tuple(
        field("cons", t.cons),
        field("v", t.v),
        field("cfl", t.cfl),
        field("dx", t.dx),
        field("L", t.L)
    );
}

inline auto fields(const patch_truth_t& t) {
    return std::make_tuple(
        field("cons", t.cons),
        field("v", t.v),
        field("cfl", t.cfl),
        field("dx", t.dx),
        field("L", t.L)
    );
}

inline auto to_truth(const patch_t& p) -> patch_truth_t {
    auto truth = patch_truth_t{};
    truth.cons = cached_t<double, 1>(p.interior, memory::host);
    copy(truth.cons[p.interior], p.cons[p.interior]);
    truth.v = p.v;
    truth.cfl = p.cfl;
    truth.dx = p.dx;
    truth.L = p.L;
    return truth;
}

inline void from_truth(patch_t& p, patch_truth_t truth) {
    p = patch_t(space(truth.cons));
    copy(p.cons[p.interior], truth.cons[p.interior]);
    p.v = truth.v;
    p.cfl = truth.cfl;
    p.dx = truth.dx;
    p.L = truth.L;
}

// =============================================================================
// 1D Linear Advection Physics Module with Domain Decomposition
// =============================================================================

// =============================================================================
// Patch key type for checkpoint filenames
// =============================================================================

struct patch_key_t {
    int start = 0;
    unsigned int size = 0;

    auto operator==(const patch_key_t&) const -> bool = default;
};

inline auto to_string(const patch_key_t& k) -> std::string {
    return std::to_string(k.start) + "_" + std::to_string(k.size);
}

inline auto from_string(std::type_identity<patch_key_t>, std::string_view sv) -> patch_key_t {
    auto s = std::string(sv);
    auto pos = s.find('_');
    return {std::stoi(s.substr(0, pos)), static_cast<unsigned int>(std::stoul(s.substr(pos + 1)))};
}

template<>
struct std::hash<patch_key_t> {
    auto operator()(const patch_key_t& k) const noexcept -> std::size_t {
        return std::hash<int>{}(k.start) ^ (std::hash<unsigned int>{}(k.size) << 1);
    }
};

// =============================================================================
// 1D Linear Advection Physics Module with Domain Decomposition
// =============================================================================

struct advection {

    struct config_t {
        int rk_order = 1;
        double cfl = 0.4;
        double wavespeed = 1.0;
        bool use_flux_buffer = false;
    };

    struct initial_t {
        unsigned int num_zones = 200;
        unsigned int num_partitions = 4;
        double domain_length = 1.0;
    };

    struct state_t {
        using patch_key_type = patch_key_t;

        double time = 0.0;
        std::vector<patch_t> patches;
    };

    using product_t = std::vector<cached_t<double, 1>>;
};

// =============================================================================
// ADL fields() functions for advection types
// =============================================================================

inline auto fields(const advection::config_t& c) {
    return std::make_tuple(
        field("rk_order", c.rk_order),
        field("cfl", c.cfl),
        field("wavespeed", c.wavespeed),
        field("use_flux_buffer", c.use_flux_buffer)
    );
}

inline auto fields(advection::config_t& c) {
    return std::make_tuple(
        field("rk_order", c.rk_order),
        field("cfl", c.cfl),
        field("wavespeed", c.wavespeed),
        field("use_flux_buffer", c.use_flux_buffer)
    );
}

inline auto fields(const advection::initial_t& i) {
    return std::make_tuple(
        field("num_zones", i.num_zones),
        field("num_partitions", i.num_partitions),
        field("domain_length", i.domain_length)
    );
}

inline auto fields(advection::initial_t& i) {
    return std::make_tuple(
        field("num_zones", i.num_zones),
        field("num_partitions", i.num_partitions),
        field("domain_length", i.domain_length)
    );
}

// =============================================================================
// Truthful implementation for advection::state_t (for single-file IO)
// =============================================================================

struct state_truth_t {
    double time = 0.0;
    std::vector<patch_truth_t> patches;
};

inline auto fields(state_truth_t& t) {
    return std::make_tuple(
        field("time", t.time),
        field("patches", t.patches)
    );
}

inline auto fields(const state_truth_t& t) {
    return std::make_tuple(
        field("time", t.time),
        field("patches", t.patches)
    );
}

inline auto to_truth(const advection::state_t& s) -> state_truth_t {
    auto truth = state_truth_t{};
    truth.time = s.time;
    for (const auto& p : s.patches) {
        truth.patches.push_back(to_truth(p));
    }
    return truth;
}

inline void from_truth(advection::state_t& s, state_truth_t truth) {
    s.time = truth.time;
    s.patches.clear();
    for (auto& pt : truth.patches) {
        patch_t p;
        from_truth(p, std::move(pt));
        s.patches.push_back(std::move(p));
    }
}

// =============================================================================
// Checkpointable protocol for advection::state_t (for parallel IO)
// =============================================================================

// Helper to get patch key from a patch
inline auto get_patch_key(const patch_t& p) -> patch_key_t {
    return {start(p.interior)[0], shape(p.interior)[0]};
}

// Emittable: emit/load header data (time only, patches written separately)
template<Sink S>
void emit(S& sink, const advection::state_t& s) {
    write(sink, "time", s.time);
}

template<Source S>
void load(S& source, advection::state_t& s) {
    read(source, "time", s.time);
    s.patches.clear();  // patches will be loaded via emplace_patch
}

// Scatterable: iterate patches and get patch data
inline auto patch_keys(const advection::state_t& s) {
    auto keys = std::vector<patch_key_t>{};
    for (const auto& p : s.patches) {
        keys.push_back(get_patch_key(p));
    }
    return keys;
}

inline auto patch_data(const advection::state_t& s, const patch_key_t& key) -> const patch_t& {
    for (const auto& p : s.patches) {
        if (get_patch_key(p) == key) {
            return p;
        }
    }
    throw std::runtime_error("patch not found: " + to_string(key));
}

// Gatherable: determine affinity and emplace patches
inline auto patch_affinity(const advection::state_t& /* s */, const patch_key_t& key, int rank, int num_ranks) -> bool {
    // Simple round-robin assignment based on patch start position
    auto hash = std::hash<patch_key_t>{}(key);
    return static_cast<int>(hash % num_ranks) == rank;
}

template<Source S>
void emplace_patch(advection::state_t& s, const patch_key_t& /* key */, S& source) {
    patch_t p;
    read(source, p);
    s.patches.push_back(std::move(p));
}

// =============================================================================
// Physics interface implementation
// =============================================================================

auto default_physics_config(std::type_identity<advection>) -> advection::config_t {
    return {.rk_order = 1, .cfl = 0.4, .wavespeed = 1.0};
}

auto default_initial_config(std::type_identity<advection>) -> advection::initial_t {
    return {.num_zones = 200, .num_partitions = 4, .domain_length = 1.0};
}

auto initial_state(
    const advection::config_t& config,
    const advection::initial_t& initial,
    const exec_context_t& ctx
) -> advection::state_t {
    auto np = initial.num_partitions;
    auto S = index_space(ivec(0), uvec(initial.num_zones));
    auto dx = initial.domain_length / initial.num_zones;
    auto L = initial.domain_length;
    auto v = config.wavespeed;
    auto cfl = config.cfl;

    auto make_patch = [&](const auto& space) {
        auto patch = patch_t(space);
        patch.dx = dx;
        patch.L = L;
        patch.v = v;
        patch.cfl = cfl;
        return patch;
    };

    auto patches = decomposed_uniform_grid(S, uvec(np), make_patch, ctx.comm);

    parallel::execute(initial_state_t{}, patches, ctx.scheduler, ctx.profiler);

    return {.time = 0.0, .patches = std::move(patches)};
}

void advance(advection::state_t& state, const exec_context_t& ctx, double dt_max) {
    // Patches store their own v, cfl, dx values from initial_state
    // Note: use_flux_buffer not accessible here, always use fused path
    auto pipeline = parallel::pipeline(
        ghost_exchange_t{},
        compute_local_dt_t{dt_max},
        global_dt_t{},
        flux_and_update_t{}
    );

    if (ctx.comm) {
        parallel::execute(pipeline, state.patches, *ctx.comm, ctx.scheduler, ctx.profiler);
    } else {
        parallel::execute(pipeline, state.patches, ctx.scheduler, ctx.profiler);
    }

    // Update time - get dt from local patch or reduce across ranks
    auto local_dt = state.patches.empty() ? 0.0 : state.patches[0].dt;
    auto dt = ctx.comm
        ? ctx.comm->combine(local_dt, [](double a, double b) { return std::max(a, b); })
        : local_dt;
    state.time += dt;
}

auto zone_count(const advection::state_t& state) -> std::size_t {
    auto count = std::size_t{0};
    for (const auto& p : state.patches) {
        count += mist::size(p.interior);
    }
    return count;
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
    const exec_context_t& /* ctx */
) -> double {
    using std::views::transform;

    if (name == "time")
        return state.time;

    // Get dx from first patch (all patches have same dx)
    const auto dx = state.patches.empty() ? 1.0 : state.patches[0].dx;
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
    const exec_context_t& /* ctx */
) -> advection::product_t {
    using std::views::transform;

    if (name == "concentration") {
        return to_vector(state.patches | transform([](const auto& p) {
            auto result = cached_t<double, 1>(p.interior, memory::host);
            copy(result[p.interior], p.cons[p.interior]);
            return result;
        }));
    }
    if (name == "cell_x") {
        return to_vector(state.patches | transform([](const auto& p) {
            auto dx = p.dx;
            return cache(lazy(p.interior, [dx](auto idx) { return cell_center_x(idx[0], dx); }), memory::host, exec::cpu);
        }));
    }
    if (name == "wavespeed") {
        return to_vector(state.patches | transform([](const auto& p) {
            auto v = p.v;
            return cache(fill(p.interior, v), memory::host, exec::cpu);
        }));
    }
    throw std::runtime_error("unknown product: " + name);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[])
{
    auto mpi = mpi_context{argc, argv};
    auto comm = mpi.create_communicator();

    auto use_socket = false;
    const char* log_prefix = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--socket") == 0 || std::strcmp(argv[i], "-s") == 0) {
            use_socket = true;
        } else if (std::strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            log_prefix = argv[++i];
        }
    }

    auto physics = mist::driver::make_physics<advection>();
    auto state = mist::driver::state_t{};
    auto engine = mist::driver::engine_t{state, *physics, std::move(comm)};

    if (log_prefix) {
        engine.set_log_prefix(log_prefix);
    }

    if (engine.get_comm().rank() == 0) {
        if (use_socket) {
            auto session = mist::driver::socket_session_t{engine};
            session.run();
        } else {
            auto session = mist::driver::repl_session_t{engine};
            session.run();
        }
    } else {
        engine.run_as_follower();
    }
    return 0;
}
