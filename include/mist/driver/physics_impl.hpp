#pragma once

#include <type_traits>
#include "physics_interface.hpp"
#include "../serialize.hpp"

namespace mist::driver {

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
            auto writer = ascii_writer{os};
            serialize(writer, "physics", config_);
        } else {
            auto writer = binary_writer{os};
            serialize(writer, "physics", config_);
        }
    }

    void write_initial(std::ostream& os, output_format fmt) override {
        if (fmt == output_format::ascii) {
            auto writer = ascii_writer{os};
            serialize(writer, "initial", initial_);
        } else {
            auto writer = binary_writer{os};
            serialize(writer, "initial", initial_);
        }
    }

    void write_state(std::ostream& os, output_format fmt) override {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        if (fmt == output_format::ascii) {
            auto writer = ascii_writer{os};
            serialize_state_to(writer);
        } else {
            auto writer = binary_writer{os};
            serialize_state_to(writer);
        }
    }

    template<typename Writer>
    void serialize_state_to(Writer& writer) {
        if (!state_.has_value()) {
            throw std::runtime_error("state not initialized");
        }
        serialize(writer, "physics", config_);
        serialize(writer, "initial", initial_);
        serialize(writer, "physics_state", *state_);
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
            auto writer = ascii_writer{os};
            for (const auto& name : selected) {
                auto product = get_product(*state_, name, *exec_context_);
                serialize(writer, name.c_str(), product);
            }
        } else {
            auto writer = binary_writer{os};
            for (const auto& name : selected) {
                auto product = get_product(*state_, name, *exec_context_);
                serialize(writer, name.c_str(), product);
            }
        }
    }

    // -------------------------------------------------------------------------
    // I/O - read operations
    // -------------------------------------------------------------------------

    auto load_physics(std::istream& is, output_format fmt) -> bool override {
        try {
            if (fmt == output_format::ascii) {
                auto reader = ascii_reader{is};
                return deserialize(reader, "physics", config_);
            } else {
                auto reader = binary_reader{is};
                return deserialize(reader, "physics", config_);
            }
        } catch (...) {
            return false;
        }
    }

    auto load_initial(std::istream& is, output_format fmt) -> bool override {
        try {
            if (fmt == output_format::ascii) {
                auto reader = ascii_reader{is};
                return deserialize(reader, "initial", initial_);
            } else {
                auto reader = binary_reader{is};
                return deserialize(reader, "initial", initial_);
            }
        } catch (...) {
            return false;
        }
    }

    auto load_state(std::istream& is, output_format fmt) -> bool override {
        try {
            if (fmt == output_format::ascii) {
                auto reader = ascii_reader{is};
                return deserialize_state_from(reader);
            } else {
                auto reader = binary_reader{is};
                return deserialize_state_from(reader);
            }
        } catch (...) {
            return false;
        }
    }

    template<typename Reader>
    auto deserialize_state_from(Reader& reader) -> bool {
        typename P::config_t cfg;
        typename P::initial_t ini;
        typename P::state_t state;

        if (!deserialize(reader, "physics", cfg)) return false;
        if (!deserialize(reader, "initial", ini)) return false;
        if (!deserialize(reader, "physics_state", state)) return false;

        config_ = std::move(cfg);
        initial_ = std::move(ini);
        state_ = std::move(state);
        return true;
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
