#pragma once

#include <filesystem>
#include <type_traits>
#include "physics_interface.hpp"
#include "../archive.hpp"

namespace mist::driver {

namespace fs = std::filesystem;

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
        : config_(default_physics_config(std::type_identity<P>{}))
        , initial_(default_initial_config(std::type_identity<P>{}))
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
        if (state_.has_value()) {
            throw std::runtime_error("state already initialized");
        }
        if (!exec_context_) {
            throw std::runtime_error("exec_context not set; engine should set it before init");
        }
        state_ = initial_state(config_, initial_, *exec_context_);
    }

    void reset() override {
        state_ = std::nullopt;
    }

    auto has_state() const -> bool override {
        return state_.has_value();
    }

    // -------------------------------------------------------------------------
    // Stepping
    // -------------------------------------------------------------------------

    void advance(double dt_max = std::numeric_limits<double>::infinity()) override {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context_) {
            throw std::runtime_error("exec_context not set");
        }
        adl_advance(*state_, *exec_context_, dt_max);
    }

    auto get_time(const std::string& var) const -> double override {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        return adl_get_time(*state_, var);
    }

    auto get_timeseries(const std::string& name) const -> double override {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context_) {
            throw std::runtime_error("exec_context not set");
        }
        return adl_get_timeseries(*state_, name, *exec_context_);
    }

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    void set_physics(const std::string& key, const std::string& value) override {
        set(config_, key, value);
    }

    void set_initial(const std::string& key, const std::string& value) override {
        if (state_.has_value()) {
            throw std::runtime_error("cannot modify initial when state exists");
        }
        set(initial_, key, value);
    }

    void set_exec_context(exec_context_t& ctx) override {
        exec_context_ = &ctx;
    }

    // -------------------------------------------------------------------------
    // I/O - write operations
    // -------------------------------------------------------------------------

    void write_physics(std::ostream& os, output_format fmt) override {
        if (fmt == output_format::ascii) {
            auto sink = ascii_sink{os};
            write(sink, "physics", config_);
        } else {
            auto sink = binary_sink{os};
            write(sink, "physics", config_);
        }
    }

    void write_initial(std::ostream& os, output_format fmt) override {
        if (fmt == output_format::ascii) {
            auto sink = ascii_sink{os};
            write(sink, "initial", initial_);
        } else {
            auto sink = binary_sink{os};
            write(sink, "initial", initial_);
        }
    }

    void write_state(std::ostream& os, output_format fmt) override {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (fmt == output_format::ascii) {
            auto sink = ascii_sink{os};
            write_state_to(sink);
        } else {
            auto sink = binary_sink{os};
            write_state_to(sink);
        }
    }

    template<typename SinkT>
    void write_state_to(SinkT& sink) {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        write(sink, "physics", config_);
        write(sink, "initial", initial_);
        write(sink, "physics_state", *state_);
    }

    void write_products(std::ostream& os, output_format fmt,
                       const std::vector<std::string>& selected) override {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context_) {
            throw std::runtime_error("exec_context not set");
        }
        if (fmt == output_format::ascii) {
            auto sink = ascii_sink{os};
            for (const auto& name : selected) {
                auto product = get_product(*state_, name, *exec_context_);
                write(sink, name.c_str(), product);
            }
        } else {
            auto sink = binary_sink{os};
            for (const auto& name : selected) {
                auto product = get_product(*state_, name, *exec_context_);
                write(sink, name.c_str(), product);
            }
        }
    }

    // -------------------------------------------------------------------------
    // I/O - read operations
    // -------------------------------------------------------------------------

    auto load_physics(std::istream& is, output_format fmt) -> bool override {
        try {
            if (fmt == output_format::ascii) {
                auto source = ascii_source{is};
                return read(source, "physics", config_);
            } else {
                auto source = binary_source{is};
                return read(source, "physics", config_);
            }
        } catch (...) {
            return false;
        }
    }

    auto load_initial(std::istream& is, output_format fmt) -> bool override {
        try {
            if (fmt == output_format::ascii) {
                auto source = ascii_source{is};
                return read(source, "initial", initial_);
            } else {
                auto source = binary_source{is};
                return read(source, "initial", initial_);
            }
        } catch (...) {
            return false;
        }
    }

    auto load_state(std::istream& is, output_format fmt) -> bool override {
        try {
            if (fmt == output_format::ascii) {
                auto source = ascii_source{is};
                return read_state_from(source);
            } else {
                auto source = binary_source{is};
                return read_state_from(source);
            }
        } catch (...) {
            return false;
        }
    }

    template<typename SourceT>
    auto read_state_from(SourceT& source) -> bool {
        typename P::config_t cfg;
        typename P::initial_t ini;
        typename P::state_t state;

        if (!read(source, "physics", cfg)) return false;
        if (!read(source, "initial", ini)) return false;
        if (!read(source, "physics_state", state)) return false;

        config_ = std::move(cfg);
        initial_ = std::move(ini);
        state_ = std::move(state);
        return true;
    }

    // -------------------------------------------------------------------------
    // Parallel I/O - directory-based operations
    // -------------------------------------------------------------------------

    void write_state(const fs::path& path, output_format fmt) override {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context_) {
            throw std::runtime_error("exec_context not set");
        }

        auto is_distributed = exec_context_->comm && exec_context_->comm->size() > 1;

        if (is_distributed) {
            if constexpr (ParallelPhysics<P>) {
                auto cp = checkpoint_t<typename P::config_t, typename P::initial_t, typename P::state_t>{
                    config_, initial_, *state_
                };
                if (fmt == output_format::binary) {
                    write_checkpoint(path, cp, Binary{});
                } else {
                    write_checkpoint(path, cp, Ascii{});
                }
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
        if (!exec_context_) {
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

                    auto rank = exec_context_->comm ? exec_context_->comm->rank() : 0;
                    auto num_ranks = exec_context_->comm ? exec_context_->comm->size() : 1;

                    if (fmt == output_format::binary) {
                        read_checkpoint(path, cp, rank, num_ranks, Binary{});
                    } else {
                        read_checkpoint(path, cp, rank, num_ranks, Ascii{});
                    }

                    config_ = std::move(cfg);
                    initial_ = std::move(ini);
                    state_ = std::move(state);
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
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (!exec_context_) {
            throw std::runtime_error("exec_context not set");
        }

        // For now, products use single-file IO only
        // TODO: Add product checkpoint support if needed
        auto mode = fmt == output_format::binary ? std::ios::binary : std::ios::out;
        std::ofstream file(path, mode);
        if (!file) {
            throw std::runtime_error("failed to open products file for writing");
        }
        write_products(file, fmt, selected);
    }

    // -------------------------------------------------------------------------
    // Profiler and performance
    // -------------------------------------------------------------------------

    auto zone_count() const -> std::size_t override {
        if (!state_.has_value()) {
            return 0;
        }
        return adl_zone_count(*state_);
    }

    auto profiler_data() const -> std::map<std::string, perf::profile_entry_t> override {
        if (!exec_context_) {
            return {};
        }
        return exec_context_->profiler.data();
    }

private:
    typename P::config_t config_;
    typename P::initial_t initial_;
    std::optional<typename P::state_t> state_;
    exec_context_t* exec_context_ = nullptr;
};

// =============================================================================
// Factory function to create a physics_impl_t
// =============================================================================

template<Physics P>
auto make_physics() -> std::unique_ptr<physics_interface_t> {
    return std::make_unique<physics_impl_t<P>>();
}

} // namespace mist::driver
