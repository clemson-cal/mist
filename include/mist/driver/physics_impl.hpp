#pragma once

#include <algorithm>
#include <filesystem>
#include <type_traits>
#include "physics_interface.hpp"
#include "../archive.hpp"

namespace mist::driver {

namespace fs = std::filesystem;

inline auto to_archive_format(output_format fmt) -> archive::format {
    return fmt == output_format::binary ? archive::format::binary : archive::format::ascii;
}

// =============================================================================
// Physics concept (defines what a physics module must provide)
// =============================================================================

// =============================================================================
// ParallelPhysics concept - physics that supports checkpoint-based parallel IO
// =============================================================================

template<typename P>
concept ParallelPhysics = requires(
    const typename P::state_t& cs,
    typename P::state_t& s,
    binary_sink& sink,
    binary_source& source,
    int rank,
    int num_ranks
) {
    typename P::state_t::patch_key_type;
    { emit(sink, cs) };
    { load(source, s) };
    { patch_keys(cs) };
    { patch_data(cs, std::declval<typename P::state_t::patch_key_type>()) };
    { patch_affinity(s, std::declval<typename P::state_t::patch_key_type>(), rank, num_ranks) } -> std::same_as<bool>;
    { emplace_patch(s, std::declval<typename P::state_t::patch_key_type>(), source) };
};

// =============================================================================
// Checkpoint wrapper - combines config, initial, and state for checkpointing
// =============================================================================

template<typename Config, typename Initial, typename State>
struct checkpoint_t {
    using patch_key_type = typename State::patch_key_type;

    const Config& config;
    const Initial& initial;
    const State& state;
};

template<typename Config, typename Initial, typename State>
struct mutable_checkpoint_t {
    using patch_key_type = typename State::patch_key_type;

    Config& config;
    Initial& initial;
    State& state;
};

// Emittable: emit writes config, initial, then state header
template<Sink S, typename C, typename I, typename St>
void emit(S& sink, const checkpoint_t<C, I, St>& cp) {
    write(sink, "config", cp.config);
    write(sink, "initial", cp.initial);
    emit(sink, cp.state);
}

template<Source S, typename C, typename I, typename St>
void load(S& source, mutable_checkpoint_t<C, I, St>& cp) {
    read(source, "config", cp.config);
    read(source, "initial", cp.initial);
    load(source, cp.state);
}

// Scatterable: delegate to state
template<typename C, typename I, typename St>
auto patch_keys(const checkpoint_t<C, I, St>& cp) {
    return patch_keys(cp.state);
}

template<typename C, typename I, typename St>
decltype(auto) patch_data(const checkpoint_t<C, I, St>& cp, const typename St::patch_key_type& key) {
    return patch_data(cp.state, key);
}

// Gatherable: delegate to state
template<typename C, typename I, typename St>
auto patch_affinity(const mutable_checkpoint_t<C, I, St>& cp,
                    const typename St::patch_key_type& key,
                    int rank, int num_ranks) -> bool {
    return patch_affinity(cp.state, key, rank, num_ranks);
}

template<Source S, typename C, typename I, typename St>
void emplace_patch(mutable_checkpoint_t<C, I, St>& cp,
                   const typename St::patch_key_type& key,
                   S& source) {
    emplace_patch(cp.state, key, source);
}

// PatchKey support for checkpoint wrapper
template<typename C, typename I, typename St>
auto to_string(const typename checkpoint_t<C, I, St>::patch_key_type& key) -> std::string {
    return to_string(key);
}

template<typename C, typename I, typename St>
auto from_string(std::type_identity<typename checkpoint_t<C, I, St>::patch_key_type>, std::string_view sv) {
    return from_string(std::type_identity<typename St::patch_key_type>{}, sv);
}

// =============================================================================
// Products wrapper - for parallel products output using Scatterable
// =============================================================================

template<typename Product>
struct patch_products_t {
    const std::vector<std::string>& names;
    const std::map<std::string, Product>& products;
    std::size_t patch_index;
};

template<Sink S, typename Product>
void write(S& sink, const patch_products_t<Product>& pp) {
    for (const auto& name : pp.names) {
        write(sink, name.c_str(), pp.products.at(name)[pp.patch_index]);
    }
}

template<typename State, typename Product>
struct products_data_t {
    using patch_key_type = typename State::patch_key_type;

    const State& state;
    const std::vector<std::string>& names;
    const std::map<std::string, Product>& products;
};

// emit: write product names to header
template<Sink S, typename St, typename Pr>
void emit(S& sink, const products_data_t<St, Pr>& pd) {
    write(sink, "products", pd.names);
}

// patch_keys: delegate to state
template<typename St, typename Pr>
auto patch_keys(const products_data_t<St, Pr>& pd) {
    return patch_keys(pd.state);
}

// patch_data: return struct with products for this patch index
template<typename St, typename Pr>
auto patch_data(const products_data_t<St, Pr>& pd,
                const typename St::patch_key_type& key) {
    auto keys = patch_keys(pd.state);
    auto it = std::find(keys.begin(), keys.end(), key);
    auto idx = static_cast<std::size_t>(std::distance(keys.begin(), it));
    return patch_products_t<Pr>{pd.names, pd.products, idx};
}

// =============================================================================
// Physics concept (defines what a physics module must provide)
// =============================================================================

template<typename P>
concept Physics = requires(
    typename P::config_t cfg,
    typename P::initial_t ini,
    typename P::state_t s,
    typename P::product_t p,
    const exec_context_t& ctx,
    double dt_max) {
    typename P::config_t;
    typename P::initial_t;
    typename P::state_t;
    typename P::product_t;

    { default_physics_config(std::type_identity<P>{}) } -> std::same_as<typename P::config_t>;
    { default_initial_config(std::type_identity<P>{}) } -> std::same_as<typename P::initial_t>;
    { initial_state(cfg, ini, ctx) } -> std::same_as<typename P::state_t>;
    { zone_count(s) } -> std::same_as<std::size_t>;

    { names_of_time(std::type_identity<P>{}) } -> std::same_as<std::vector<std::string>>;
    { names_of_timeseries(std::type_identity<P>{}) } -> std::same_as<std::vector<std::string>>;
    { names_of_products(std::type_identity<P>{}) } -> std::same_as<std::vector<std::string>>;

    { get_time(s, std::string{}) } -> std::same_as<double>;
    { get_timeseries(s, std::string{}, ctx) } -> std::same_as<double>;
    { get_product(s, std::string{}, ctx) } -> std::same_as<typename P::product_t>;

    { advance(s, ctx, dt_max) } -> std::same_as<void>;
};

// =============================================================================
// ADL helper functions (avoid name collision with member functions)
// =============================================================================

template<typename State>
void adl_advance(State& s, const exec_context_t& ctx, double dt_max) { advance(s, ctx, dt_max); }

template<typename State>
auto adl_get_time(const State& s, const std::string& var) -> double { return get_time(s, var); }

template<typename State>
auto adl_get_timeseries(const State& s, const std::string& name, const exec_context_t& ctx) -> double {
    return get_timeseries(s, name, ctx);
}

template<typename State>
auto adl_zone_count(const State& s) -> std::size_t { return zone_count(s); }

// =============================================================================
// physics_impl_t - type-erased implementation of physics_interface_t
// =============================================================================

template<Physics P>
class physics_impl_t : public physics_interface_t {
public:
    physics_impl_t()
        : config(default_physics_config(std::type_identity<P>{}))
        , initial(default_initial_config(std::type_identity<P>{}))
    {}

    // -------------------------------------------------------------------------
    // Discovery
    // -------------------------------------------------------------------------

    auto time_names() const -> std::vector<std::string> override {
        return names_of_time(std::type_identity<P>{});
    }

    auto timeseries_names() const -> std::vector<std::string> override {
        return names_of_timeseries(std::type_identity<P>{});
    }

    auto product_names() const -> std::vector<std::string> override {
        return names_of_products(std::type_identity<P>{});
    }

    // -------------------------------------------------------------------------
    // State management
    // -------------------------------------------------------------------------

    void init() override {
        if (state.has_value()) {
            throw std::runtime_error("state already initialized");
        }
        if (!exec_context) {
            throw std::runtime_error("exec_context not set; engine should set it before init");
        }
        state = initial_state(config, initial, *exec_context);
    }

    void reset() override {
        state = std::nullopt;
    }

    auto has_state() const -> bool override {
        return state.has_value();
    }

    // -------------------------------------------------------------------------
    // Stepping
    // -------------------------------------------------------------------------

    void advance(double dt_max = std::numeric_limits<double>::infinity()) override {
        if (!state.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context) {
            throw std::runtime_error("exec_context not set");
        }
        adl_advance(*state, *exec_context, dt_max);
    }

    auto get_time(const std::string& var) const -> double override {
        if (!state.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        return adl_get_time(*state, var);
    }

    auto get_timeseries(const std::string& name) const -> double override {
        if (!state.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context) {
            throw std::runtime_error("exec_context not set");
        }
        return adl_get_timeseries(*state, name, *exec_context);
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    void set_physics(const std::string& key, const std::string& value) override {
        set(config, key, value);
    }

    void set_initial(const std::string& key, const std::string& value) override {
        if (state.has_value()) {
            throw std::runtime_error("cannot modify initial when state exists");
        }
        set(initial, key, value);
    }

    void set_exec_context(exec_context_t& ctx) override {
        exec_context = &ctx;
    }

    // -------------------------------------------------------------------------
    // I/O - write operations
    // -------------------------------------------------------------------------

    void write_physics(std::ostream& os, output_format fmt) override {
        auto f = to_archive_format(fmt);
        archive::with_sink(os, f, [&](auto& sink) { write(sink, "physics", config); });
    }

    void write_initial(std::ostream& os, output_format fmt) override {
        auto f = to_archive_format(fmt);
        archive::with_sink(os, f, [&](auto& sink) { write(sink, "initial", initial); });
    }

    void write_state(std::ostream& os, output_format fmt) override {
        if (!state.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        auto f = to_archive_format(fmt);
        archive::with_sink(os, f, [&](auto& sink) { write_state_to(sink); });
    }

    template<typename SinkT>
    void write_state_to(SinkT& sink) {
        write(sink, "physics", config);
        write(sink, "initial", initial);
        write(sink, "physics_state", *state);
    }

    void write_products(std::ostream& os, output_format fmt,
                       const std::vector<std::string>& selected) override {
        if (!state.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context) {
            throw std::runtime_error("exec_context not set");
        }
        auto f = to_archive_format(fmt);
        archive::with_sink(os, f, [&](auto& sink) {
            for (const auto& name : selected) {
                auto product = get_product(*state, name, *exec_context);
                write(sink, name.c_str(), product);
            }
        });
    }

    // -------------------------------------------------------------------------
    // I/O - read operations
    // -------------------------------------------------------------------------

    auto load_physics(std::istream& is, output_format fmt) -> bool override {
        try {
            auto f = to_archive_format(fmt);
            return archive::with_source(is, f, [&](auto& source) {
                return read(source, "physics", config);
            });
        } catch (...) {
            return false;
        }
    }

    auto load_initial(std::istream& is, output_format fmt) -> bool override {
        try {
            auto f = to_archive_format(fmt);
            return archive::with_source(is, f, [&](auto& source) {
                return read(source, "initial", initial);
            });
        } catch (...) {
            return false;
        }
    }

    auto load_state(std::istream& is, output_format fmt) -> bool override {
        try {
            auto f = to_archive_format(fmt);
            return archive::with_source(is, f, [&](auto& source) {
                return read_statefrom(source);
            });
        } catch (...) {
            return false;
        }
    }

    template<typename SourceT>
    auto read_statefrom(SourceT& source) -> bool {
        auto cfg = typename P::config_t{};
        auto ini = typename P::initial_t{};
        auto st = typename P::state_t{};

        if (!read(source, "physics", cfg)) return false;
        if (!read(source, "initial", ini)) return false;
        if (!read(source, "physics_state", st)) return false;

        config = std::move(cfg);
        initial = std::move(ini);
        state = std::move(st);
        return true;
    }

    // -------------------------------------------------------------------------
    // Parallel I/O - directory-based operations
    // -------------------------------------------------------------------------

    void write_state(const fs::path& path, output_format fmt) override {
        if (!state.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context) {
            throw std::runtime_error("exec_context not set");
        }

        auto is_distributed = exec_context->comm && exec_context->comm->size() > 1;

        if (is_distributed) {
            if constexpr (ParallelPhysics<P>) {
                auto cp = checkpoint_t<typename P::config_t, typename P::initial_t, typename P::state_t>{
                    config, initial, *state
                };
                distributed_write(path, cp, to_archive_format(fmt));
            } else {
                throw std::runtime_error("this physics module does not support parallel IO");
            }
        } else {
            // Single-file IO for non-distributed case
            auto mode = fmt == output_format::binary ? std::ios::binary : std::ios::out;
            std::ofstream file(path, mode);
            if (!file) {
                throw std::runtime_error("failed to open state file for writing");
            }
            write_state(file, fmt);
        }
    }

    auto load_state(const fs::path& path, output_format fmt, item_predicate /* wants_item */) -> bool override {
        if (!exec_context) {
            throw std::runtime_error("exec_context not set");
        }

        // Check if path is a directory (parallel IO) or file (single-file IO)
        if (fs::is_directory(path)) {
            if constexpr (ParallelPhysics<P>) {
                try {
                    typename P::config_t cfg;
                    typename P::initial_t ini;
                    typename P::state_t state;

                    auto cp = mutable_checkpoint_t<typename P::config_t, typename P::initial_t, typename P::state_t>{
                        cfg, ini, state
                    };

                    auto rank = exec_context->comm ? exec_context->comm->rank() : 0;
                    auto num_ranks = exec_context->comm ? exec_context->comm->size() : 1;

                    distributed_read(path, cp, rank, num_ranks, to_archive_format(fmt));

                    config = std::move(cfg);
                    initial = std::move(ini);
                    state = std::move(state);
                    return true;
                } catch (...) {
                    return false;
                }
            } else {
                throw std::runtime_error("this physics module does not support parallel IO");
            }
        } else {
            // Single-file IO
            auto mode = fmt == output_format::binary ? std::ios::binary : std::ios::in;
            std::ifstream file(path, mode);
            if (!file) return false;
            return load_state(file, fmt);
        }
    }

    void write_products(const fs::path& path, output_format fmt,
                        const std::vector<std::string>& selected) override {
        if (!state.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context) {
            throw std::runtime_error("exec_context not set");
        }

        auto is_distributed = exec_context->comm && exec_context->comm->size() > 1;

        if (is_distributed) {
            if constexpr (ParallelPhysics<P>) {
                write_products_parallel(path, fmt, selected);
            } else {
                throw std::runtime_error("this physics module does not support parallel IO");
            }
        } else {
            auto mode = fmt == output_format::binary ? std::ios::binary : std::ios::out;
            auto file = std::ofstream{path, mode};
            if (!file) {
                throw std::runtime_error("failed to open products file for writing");
            }
            write_products(file, fmt, selected);
        }
    }

    void write_products_parallel(const fs::path& dir, output_format fmt,
                                 const std::vector<std::string>& selected)
        requires ParallelPhysics<P>
    {
        auto products = std::map<std::string, typename P::product_t>{};
        for (const auto& name : selected) {
            products[name] = get_product(*state, name, *exec_context);
        }

        auto pd = products_data_t<typename P::state_t, typename P::product_t>{
            *state, selected, products
        };
        distributed_write(dir, pd, to_archive_format(fmt));
    }

    // -------------------------------------------------------------------------
    // Profiler and performance
    // -------------------------------------------------------------------------

    auto zone_count() const -> std::size_t override {
        if (!state.has_value()) {
            return 0;
        }
        return adl_zone_count(*state);
    }

    auto profiler_data() const -> std::map<std::string, perf::profile_entry_t> override {
        if (!exec_context) {
            return {};
        }
        return exec_context->profiler.data();
    }

private:
    typename P::config_t config;
    typename P::initial_t initial;
    std::optional<typename P::state_t> state;
    exec_context_t* exec_context = nullptr;
};

// =============================================================================
// Factory function to create a physics_impl_t
// =============================================================================

template<Physics P>
auto make_physics() -> std::unique_ptr<physics_interface_t> {
    return std::make_unique<physics_impl_t<P>>();
}

} // namespace mist::driver
