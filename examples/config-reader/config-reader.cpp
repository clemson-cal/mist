#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include "mist/core.hpp"
#include "mist/archive.hpp"

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
    double value = 0.0;
};

inline auto fields(const boundary_t& b) {
    return std::make_tuple(field("type", b.type), field("value", b.value));
}

inline auto fields(boundary_t& b) {
    return std::make_tuple(field("type", b.type), field("value", b.value));
}

struct mesh_t {
    vec_t<int, 3> resolution;
    vec_t<double, 3> lower;
    vec_t<double, 3> upper;
    boundary_t boundary_lo;
    boundary_t boundary_hi;
};

inline auto fields(const mesh_t& m) {
    return std::make_tuple(
        field("resolution", m.resolution),
        field("lower", m.lower),
        field("upper", m.upper),
        field("boundary_lo", m.boundary_lo),
        field("boundary_hi", m.boundary_hi)
    );
}

inline auto fields(mesh_t& m) {
    return std::make_tuple(
        field("resolution", m.resolution),
        field("lower", m.lower),
        field("upper", m.upper),
        field("boundary_lo", m.boundary_lo),
        field("boundary_hi", m.boundary_hi)
    );
}

struct physics_t {
    double gamma;
    double cfl;
    std::vector<double> diffusion_coeffs;
};

inline auto fields(const physics_t& p) {
    return std::make_tuple(
        field("gamma", p.gamma),
        field("cfl", p.cfl),
        field("diffusion_coeffs", p.diffusion_coeffs)
    );
}

inline auto fields(physics_t& p) {
    return std::make_tuple(
        field("gamma", p.gamma),
        field("cfl", p.cfl),
        field("diffusion_coeffs", p.diffusion_coeffs)
    );
}

struct source_t {
    std::string name;
    vec_t<double, 3> position;
    vec_t<double, 3> velocity;
    double radius;
    double amplitude;
};

inline auto fields(const source_t& s) {
    return std::make_tuple(
        field("name", s.name),
        field("position", s.position),
        field("velocity", s.velocity),
        field("radius", s.radius),
        field("amplitude", s.amplitude)
    );
}

inline auto fields(source_t& s) {
    return std::make_tuple(
        field("name", s.name),
        field("position", s.position),
        field("velocity", s.velocity),
        field("radius", s.radius),
        field("amplitude", s.amplitude)
    );
}

struct output_t {
    std::string directory;
    std::string prefix;
    std::vector<double> snapshot_times;
    int checkpoint_interval;
    double timeseries_dt;
};

inline auto fields(const output_t& o) {
    return std::make_tuple(
        field("directory", o.directory),
        field("prefix", o.prefix),
        field("snapshot_times", o.snapshot_times),
        field("checkpoint_interval", o.checkpoint_interval),
        field("timeseries_dt", o.timeseries_dt)
    );
}

inline auto fields(output_t& o) {
    return std::make_tuple(
        field("directory", o.directory),
        field("prefix", o.prefix),
        field("snapshot_times", o.snapshot_times),
        field("checkpoint_interval", o.checkpoint_interval),
        field("timeseries_dt", o.timeseries_dt)
    );
}

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
};

inline auto fields(const config_t& c) {
    return std::make_tuple(
        field("title", c.title),
        field("description", c.description),
        field("version", c.version),
        field("t_final", c.t_final),
        field("max_iterations", c.max_iterations),
        field("mesh", c.mesh),
        field("physics", c.physics),
        field("sources", c.sources),
        field("output", c.output)
    );
}

inline auto fields(config_t& c) {
    return std::make_tuple(
        field("title", c.title),
        field("description", c.description),
        field("version", c.version),
        field("t_final", c.t_final),
        field("max_iterations", c.max_iterations),
        field("mesh", c.mesh),
        field("physics", c.physics),
        field("sources", c.sources),
        field("output", c.output)
    );
}

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
        ascii_source source(file);
        config_t config;
        read(source, "config", config);

        std::cout << "Configuration loaded successfully!\n";
        std::cout << "========================================\n\n";

        // Output the configuration using ascii_sink
        ascii_sink sink(std::cout);
        write(sink, "config", config);

        // Demo: use set() to override fields by path
        std::cout << "\n========================================\n";
        std::cout << "Demonstrating set() function:\n";
        std::cout << "========================================\n\n";

        set(config, "t_final", "20.0");
        set(config, "physics.gamma", "1.33");
        set(config, "mesh.boundary_lo.type", "reflecting");

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
