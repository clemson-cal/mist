#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include "mist/core.hpp"
#include "mist/serialize.hpp"

using namespace mist;

// =============================================================================
// Boundary condition enum with ADL string conversion
// =============================================================================

enum class boundary_condition { periodic, outflow, reflecting };

inline const char* to_string(boundary_condition bc) {
    switch (bc) {
        case boundary_condition::periodic: return "periodic";
        case boundary_condition::outflow: return "outflow";
        case boundary_condition::reflecting: return "reflecting";
    }
    return "unknown";
}

inline boundary_condition from_string(std::type_identity<boundary_condition>, const std::string& s) {
    if (s == "periodic") return boundary_condition::periodic;
    if (s == "outflow") return boundary_condition::outflow;
    if (s == "reflecting") return boundary_condition::reflecting;
    throw std::runtime_error("invalid boundary_condition: " + s);
}

// =============================================================================
// Nested configuration structures
// =============================================================================

struct boundary_t {
    boundary_condition type = boundary_condition::periodic;
    double value = 0.0;                // boundary value (if applicable)

    auto fields() const {
        return std::make_tuple(
            field("type", type),
            field("value", value)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("type", type),
            field("value", value)
        );
    }
};

struct mesh_t {
    vec_t<int, 3> resolution;
    vec_t<double, 3> lower;
    vec_t<double, 3> upper;
    boundary_t boundary_lo;
    boundary_t boundary_hi;

    auto fields() const {
        return std::make_tuple(
            field("resolution", resolution),
            field("lower", lower),
            field("upper", upper),
            field("boundary_lo", boundary_lo),
            field("boundary_hi", boundary_hi)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("resolution", resolution),
            field("lower", lower),
            field("upper", upper),
            field("boundary_lo", boundary_lo),
            field("boundary_hi", boundary_hi)
        );
    }
};

struct physics_t {
    double gamma;
    double cfl;
    std::vector<double> diffusion_coeffs;

    auto fields() const {
        return std::make_tuple(
            field("gamma", gamma),
            field("cfl", cfl),
            field("diffusion_coeffs", diffusion_coeffs)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("gamma", gamma),
            field("cfl", cfl),
            field("diffusion_coeffs", diffusion_coeffs)
        );
    }
};

struct source_t {
    std::string name;
    vec_t<double, 3> position;
    vec_t<double, 3> velocity;
    double radius;
    double amplitude;

    auto fields() const {
        return std::make_tuple(
            field("name", name),
            field("position", position),
            field("velocity", velocity),
            field("radius", radius),
            field("amplitude", amplitude)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("name", name),
            field("position", position),
            field("velocity", velocity),
            field("radius", radius),
            field("amplitude", amplitude)
        );
    }
};

struct output_t {
    std::string directory;
    std::string prefix;
    std::vector<double> snapshot_times;
    int checkpoint_interval;
    double timeseries_dt;

    auto fields() const {
        return std::make_tuple(
            field("directory", directory),
            field("prefix", prefix),
            field("snapshot_times", snapshot_times),
            field("checkpoint_interval", checkpoint_interval),
            field("timeseries_dt", timeseries_dt)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("directory", directory),
            field("prefix", prefix),
            field("snapshot_times", snapshot_times),
            field("checkpoint_interval", checkpoint_interval),
            field("timeseries_dt", timeseries_dt)
        );
    }
};

struct config_t {
    std::string title;
    std::string description;
    int version;
    double t_final;
    int max_iterations;
    mesh_t mesh;
    physics_t physics;
    std::vector<source_t> sources;
    output_t output;

    auto fields() const {
        return std::make_tuple(
            field("title", title),
            field("description", description),
            field("version", version),
            field("t_final", t_final),
            field("max_iterations", max_iterations),
            field("mesh", mesh),
            field("physics", physics),
            field("sources", sources),
            field("output", output)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("title", title),
            field("description", description),
            field("version", version),
            field("t_final", t_final),
            field("max_iterations", max_iterations),
            field("mesh", mesh),
            field("physics", physics),
            field("sources", sources),
            field("output", output)
        );
    }
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Error: cannot open file '" << argv[1] << "'\n";
        return 1;
    }

    try {
        ascii_reader reader(file);
        config_t config;
        mist::deserialize(reader, "config", config);

        std::cout << "Configuration loaded successfully!\n";
        std::cout << "========================================\n\n";

        // Output the configuration using ascii_writer
        ascii_writer writer(std::cout);
        mist::serialize(writer, "config", config);

        // Demo: use set() to override fields by path
        std::cout << "\n========================================\n";
        std::cout << "Demonstrating set() function:\n";
        std::cout << "========================================\n\n";

        mist::set(config, "t_final", "20.0");
        mist::set(config, "physics.gamma", "1.33");
        mist::set(config, "mesh.boundary_lo.type", "reflecting");

        std::cout << "After overrides:\n";
        std::cout << "  t_final = " << config.t_final << "\n";
        std::cout << "  physics.gamma = " << config.physics.gamma << "\n";
        std::cout << "  mesh.boundary_lo.type = " << to_string(config.mesh.boundary_lo.type) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error parsing config: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
