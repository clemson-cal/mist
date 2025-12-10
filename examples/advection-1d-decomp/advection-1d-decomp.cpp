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
// Ghost message
// =============================================================================

struct ghost_message_t {
    int    source_rank;
    double l_guard;
    double r_guard;
};

// =============================================================================
// Patch state - unified context that flows through pipeline
// =============================================================================

struct patch_state_t {
    int rank = 0;
    int npartitions = 1;
    int l_rank = 0;
    int r_rank = 0;
    int i0 = 0, i1 = 0;

    double v = 0.0;
    double dx = 0.0;
    double dt = 0.0;

    cached_t<double, 1> conserved;

    double l_guard = 0.0;
    double r_guard = 0.0;

    patch_state_t() = default;

    patch_state_t(int rank_, int npartitions_, cached_t<double, 1> conserved_)
        : rank(rank_)
        , npartitions(npartitions_)
        , l_rank((rank_ - 1 + npartitions_) % npartitions_)
        , r_rank((rank_ + 1) % npartitions_)
        , conserved(std::move(conserved_))
    {
        auto s = space(conserved);
        i0 = start(s)[0];
        i1 = i0 + shape(s)[0];
    }

    void set_step_params(double v_, double dx_, double dt_) {
        v = v_;
        dx = dx_;
        dt = dt_;
        l_guard = 0.0;
        r_guard = 0.0;
    }
};

// =============================================================================
// Stage 1: Ghost exchange (Message stage)
// =============================================================================

class ghost_exchange_t {
public:
    using key_t = int;
    using message_t = ghost_message_t;
    using context_t = patch_state_t;

    ghost_exchange_t(patch_state_t p) : patch(std::move(p)) {}

    auto key() const -> key_t { return patch.rank; }

    void messages(auto send) {
        auto l_val = patch.conserved(ivec(patch.i0));
        auto r_val = patch.conserved(ivec(patch.i1 - 1));
        send(patch.l_rank, ghost_message_t{patch.rank, 0.0, l_val});
        send(patch.r_rank, ghost_message_t{patch.rank, r_val, 0.0});
    }

    auto receive(message_t msg) -> std::optional<context_t> {
        if (msg.source_rank == patch.l_rank) {
            patch.l_guard = msg.l_guard;
            received++;
        } else if (msg.source_rank == patch.r_rank) {
            patch.r_guard = msg.r_guard;
            received++;
        }
        if (received == 2) {
            return std::move(patch);
        }
        return std::nullopt;
    }

private:
    patch_state_t patch;
    int received = 0;
};

// =============================================================================
// Stage 2: Flux and update (Compute stage)
// =============================================================================

class flux_and_update_t {
public:
    using context_t = patch_state_t;

    flux_and_update_t(patch_state_t p) : patch(std::move(p)) {}

    auto value() && -> context_t {
        if (patch.v > 0) {
            for (auto i = patch.i1 - 1; i >= patch.i0; --i) {
                auto u_l = (i == patch.i0) ? patch.l_guard : patch.conserved[i - 1];
                auto flux_l = patch.v * u_l;
                auto flux_r = patch.v * patch.conserved[i];
                patch.conserved[i] = patch.conserved[i] - patch.dt / patch.dx * (flux_r - flux_l);
            }
        } else {
            for (auto i = patch.i0; i < patch.i1; ++i) {
                auto u_r = (i == patch.i1 - 1) ? patch.r_guard : patch.conserved[i + 1];
                auto flux_l = patch.v * patch.conserved[i];
                auto flux_r = patch.v * u_r;
                patch.conserved[i] = patch.conserved[i] - patch.dt / patch.dx * (flux_r - flux_l);
            }
        }
        return std::move(patch);
    }

private:
    patch_state_t patch;
};

// =============================================================================
// Custom serialization for patch_state_t
// =============================================================================

template<ArchiveWriter A>
void serialize(A& ar, const patch_state_t& patch) {
    ar.begin_group();
    serialize(ar, "rank", patch.rank);
    serialize(ar, "npartitions", patch.npartitions);
    serialize(ar, "conserved", patch.conserved);
    ar.end_group();
}

template<ArchiveReader A>
auto deserialize(A& ar, patch_state_t& patch) -> bool {
    if (!ar.begin_group()) return false;
    auto rank = 0;
    auto npartitions = 1;
    auto conserved = cached_t<double, 1>{};
    deserialize(ar, "rank", rank);
    deserialize(ar, "npartitions", npartitions);
    deserialize(ar, "conserved", conserved);
    ar.end_group();
    patch = patch_state_t(rank, npartitions, std::move(conserved));
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
        mutable std::unique_ptr<parallel::thread_pool_t> pool;

        exec_context_t(const config_t& cfg, const initial_t& ini)
            : config(cfg), initial(ini), pool(std::make_unique<parallel::thread_pool_t>(4)) {}

        void set_num_threads(std::size_t n) {
            pool = std::make_unique<parallel::thread_pool_t>(n);
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
    auto global_space = index_space(ivec(0), uvec(ini.num_zones));
    auto dx = ini.domain_length / ini.num_zones;
    auto L = ini.domain_length;

    auto patches = to_vector(iota(0, npartitions) | transform([&](int p) {
        auto s = subspace(global_space, npartitions, p, 0);
        auto u = lazy(s, [&](auto i) {
            return std::sin(2.0 * M_PI * cell_center_x(i[0], dx) / L);
        });
        return patch_state_t(p, npartitions, cache(u, memory::host, exec::cpu));
    }));
    return {std::move(patches), 0.0};
}

void advance(
    advection::state_t& state,
    double dt,
    const advection::exec_context_t& exec
) {
    if (exec.config.rk_order != 1) {
        throw std::runtime_error("only rk_order=1 (forward Euler) is supported");
    }

    auto npartitions = static_cast<int>(state.patches.size());
    auto dx = exec.initial.domain_length / exec.initial.num_zones;
    auto v = exec.config.advection_velocity;

    // Set step parameters on all patches
    for (auto& patch : state.patches) {
        patch.set_step_params(v, dx, dt);
    }

    // Execute pipeline in-place on patches vector
    using advection_pipeline = parallel::pipeline<patch_state_t, ghost_exchange_t, flux_and_update_t>;
    auto comm = parallel::local_communicator<ghost_message_t>(npartitions);
    parallel::execute(advection_pipeline{}, state.patches, comm, *exec.pool);

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
