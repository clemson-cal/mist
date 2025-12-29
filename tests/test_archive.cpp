#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include "mist/archive/archive.hpp"

// =============================================================================
// Test structures
// =============================================================================

struct particle_t {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    double mass;
};

auto fields(particle_t& p) {
    return std::make_tuple(
        archive::field("position", p.position),
        archive::field("velocity", p.velocity),
        archive::field("mass", p.mass)
    );
}

auto fields(const particle_t& p) {
    return std::make_tuple(
        archive::field("position", p.position),
        archive::field("velocity", p.velocity),
        archive::field("mass", p.mass)
    );
}

struct grid_config_t {
    std::array<int, 3> resolution;
    std::array<double, 3> domain_min;
    std::array<double, 3> domain_max;
};

auto fields(grid_config_t& g) {
    return std::make_tuple(
        archive::field("resolution", g.resolution),
        archive::field("domain_min", g.domain_min),
        archive::field("domain_max", g.domain_max)
    );
}

auto fields(const grid_config_t& g) {
    return std::make_tuple(
        archive::field("resolution", g.resolution),
        archive::field("domain_min", g.domain_min),
        archive::field("domain_max", g.domain_max)
    );
}

struct simulation_state_t {
    double time;
    int iteration;
    grid_config_t grid;
    std::vector<particle_t> particles;
    std::vector<double> scalar_field;
};

auto fields(simulation_state_t& s) {
    return std::make_tuple(
        archive::field("time", s.time),
        archive::field("iteration", s.iteration),
        archive::field("grid", s.grid),
        archive::field("particles", s.particles),
        archive::field("scalar_field", s.scalar_field)
    );
}

auto fields(const simulation_state_t& s) {
    return std::make_tuple(
        archive::field("time", s.time),
        archive::field("iteration", s.iteration),
        archive::field("grid", s.grid),
        archive::field("particles", s.particles),
        archive::field("scalar_field", s.scalar_field)
    );
}

struct nested_strings_t {
    std::vector<std::string> items;
};

auto fields(nested_strings_t& n) {
    return std::make_tuple(archive::field("items", n.items));
}

auto fields(const nested_strings_t& n) {
    return std::make_tuple(archive::field("items", n.items));
}

// =============================================================================
// Equality comparators
// =============================================================================

auto approx_equal(double a, double b, double tol = 1e-10) -> bool {
    return std::abs(a - b) < tol;
}

template<typename T, std::size_t N>
auto equal(const std::array<T, N>& a, const std::array<T, N>& b) -> bool {
    for (auto i = std::size_t{0}; i < N; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!approx_equal(a[i], b[i])) return false;
        } else {
            if (a[i] != b[i]) return false;
        }
    }
    return true;
}

auto equal(int a, int b) -> bool {
    return a == b;
}

auto equal(double a, double b) -> bool {
    return approx_equal(a, b);
}

auto equal(const std::string& a, const std::string& b) -> bool {
    return a == b;
}

template<typename T>
auto equal(const std::vector<T>& a, const std::vector<T>& b) -> bool {
    if (a.size() != b.size()) return false;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
        if (!equal(a[i], b[i])) return false;
    }
    return true;
}

auto equal(const particle_t& a, const particle_t& b) -> bool {
    return equal(a.position, b.position) &&
           equal(a.velocity, b.velocity) &&
           equal(a.mass, b.mass);
}

auto equal(const grid_config_t& a, const grid_config_t& b) -> bool {
    return equal(a.resolution, b.resolution) &&
           equal(a.domain_min, b.domain_min) &&
           equal(a.domain_max, b.domain_max);
}

auto equal(const simulation_state_t& a, const simulation_state_t& b) -> bool {
    return equal(a.time, b.time) &&
           equal(a.iteration, b.iteration) &&
           equal(a.grid, b.grid) &&
           equal(a.particles, b.particles) &&
           equal(a.scalar_field, b.scalar_field);
}

auto equal(const nested_strings_t& a, const nested_strings_t& b) -> bool {
    return equal(a.items, b.items);
}

// =============================================================================
// Test data factories
// =============================================================================

auto make_test_int() -> int {
    return 42;
}

auto make_test_double() -> double {
    return 3.14159265358979;
}

auto make_test_string() -> std::string {
    return "Hello, World!\nWith escape chars: \t\"quoted\"";
}

auto make_test_vec3_double() -> std::array<double, 3> {
    return {1.5, 2.5, 3.5};
}

auto make_test_vec3_int() -> std::array<int, 3> {
    return {64, 128, 256};
}

auto make_test_vector_double() -> std::vector<double> {
    return {300.0, 305.2, 298.5, 302.1};
}

auto make_test_vector_double_empty() -> std::vector<double> {
    return {};
}

auto make_test_vector_double_large() -> std::vector<double> {
    auto result = std::vector<double>{};
    for (auto i = 0; i < 1000; ++i) {
        result.push_back(static_cast<double>(i) * 0.001);
    }
    return result;
}

auto make_test_vector_string() -> std::vector<std::string> {
    return {"default", "primitive", "conserved"};
}

auto make_test_vector_string_empty() -> std::vector<std::string> {
    return {};
}

auto make_test_particle() -> particle_t {
    return {{{0.1, 0.2, 0.15}}, {{1.5, -0.3, 0.0}}, 1.2};
}

auto make_test_vector_particle() -> std::vector<particle_t> {
    return {
        {{{0.1, 0.2, 0.15}}, {{1.5, -0.3, 0.0}}, 1.0},
        {{{0.8, 0.7, 0.25}}, {{-0.5, 0.8, 0.2}}, 2.0}
    };
}

auto make_test_grid_config() -> grid_config_t {
    return {{{64, 64, 32}}, {{0.0, 0.0, 0.0}}, {{1.0, 1.0, 0.5}}};
}

auto make_test_simulation_state() -> simulation_state_t {
    auto state = simulation_state_t{};
    state.time = 1.234;
    state.iteration = 42;
    state.grid = make_test_grid_config();
    state.particles = make_test_vector_particle();
    state.scalar_field = make_test_vector_double();
    return state;
}

auto make_test_nested_strings() -> nested_strings_t {
    return {{"alpha", "beta", "gamma"}};
}

auto make_test_nested_strings_empty() -> nested_strings_t {
    return {{}};
}

// =============================================================================
// Format tags and round-trip
// =============================================================================

struct ascii_tag {
    static auto make_stream() -> std::stringstream {
        return std::stringstream{};
    }

    static auto make_sink(std::stringstream& ss) -> archive::ascii_sink {
        return archive::ascii_sink(ss);
    }

    static auto make_source(std::stringstream& ss) -> archive::ascii_source {
        return archive::ascii_source(ss);
    }
};

struct binary_tag {
    static auto make_stream() -> std::stringstream {
        return std::stringstream(std::ios::binary | std::ios::in | std::ios::out);
    }

    static auto make_sink(std::stringstream& ss) -> archive::binary_sink {
        return archive::binary_sink(ss);
    }

    static auto make_source(std::stringstream& ss) -> archive::binary_source {
        return archive::binary_source(ss);
    }
};

struct binary_file_tag {
    static constexpr const char* filename = "test_round_trip.bin";
};

// Default construction for most types
template<typename T>
auto make_default() -> T {
    return T{};
}

template<typename Format, typename T>
auto round_trip(const T& original, const char* name) -> T {
    auto ss = Format::make_stream();
    auto sink = Format::make_sink(ss);
    archive::write(sink, name, original);

    ss.seekg(0);
    auto source = Format::make_source(ss);
    auto loaded = make_default<T>();
    archive::read(source, name, loaded);
    return loaded;
}

template<typename T>
auto round_trip_file(const T& original, const char* name) -> T {
    {
        auto file = std::ofstream(binary_file_tag::filename, std::ios::binary);
        auto sink = archive::binary_sink(file);
        archive::write(sink, name, original);
    }

    auto loaded = make_default<T>();
    {
        auto file = std::ifstream(binary_file_tag::filename, std::ios::binary);
        auto source = archive::binary_source(file);
        archive::read(source, name, loaded);
    }
    std::remove(binary_file_tag::filename);
    return loaded;
}

// =============================================================================
// Test driver
// =============================================================================

template<typename Format, typename Factory>
void test_format(Factory make_value, const char* field_name) {
    auto original = make_value();
    auto loaded = round_trip<Format>(original, field_name);
    assert(equal(original, loaded));
}

template<typename Factory>
void test_format_file(Factory make_value, const char* field_name) {
    auto original = make_value();
    auto loaded = round_trip_file(original, field_name);
    assert(equal(original, loaded));
}

template<typename Factory>
void test_round_trip(const char* type_name, Factory make_value, const char* field_name) {
    std::cout << "Testing " << type_name << "... ";
    test_format<ascii_tag>(make_value, field_name);
    test_format<binary_tag>(make_value, field_name);
    test_format_file(make_value, field_name);
    std::cout << "PASSED\n";
}

// =============================================================================
// Additional tests (size comparison, file I/O)
// =============================================================================

void test_binary_vs_ascii_size() {
    std::cout << "Testing binary vs ASCII size comparison... ";

    auto state = simulation_state_t{};
    state.time = 1.234;
    state.iteration = 42;
    state.grid = make_test_grid_config();
    state.particles = make_test_vector_particle();

    for (auto i = 0; i < 1000000; ++i) {
        state.scalar_field.push_back(300.0 + 0.00001 * i);
    }

    auto ascii_ss = std::stringstream{};
    auto ascii_sink = archive::ascii_sink(ascii_ss);
    archive::write(ascii_sink, "state", state);
    auto ascii_size = ascii_ss.str().size();

    auto binary_ss = std::stringstream(std::ios::binary | std::ios::in | std::ios::out);
    auto binary_sink = archive::binary_sink(binary_ss);
    archive::write(binary_sink, "state", state);
    auto binary_size = binary_ss.str().size();

    std::cout << "\n    ASCII size:  " << ascii_size << " bytes\n";
    std::cout << "    Binary size: " << binary_size << " bytes\n";
    std::cout << "    Ratio: " << static_cast<double>(binary_size) / ascii_size * 100.0 << "%\n";

    assert(binary_size < ascii_size);

    std::cout << "    PASSED\n";
}

void test_binary_file_io() {
    std::cout << "Testing binary file I/O... ";

    auto original = make_test_vector_double_large();
    {
        auto file = std::ofstream("test_output.bin", std::ios::binary);
        auto sink = archive::binary_sink(file);
        archive::write(sink, "data", original);
    }

    auto loaded = std::vector<double>{};
    {
        auto file = std::ifstream("test_output.bin", std::ios::binary);
        auto source = archive::binary_source(file);
        archive::read(source, "data", loaded);
    }

    assert(equal(original, loaded));
    std::remove("test_output.bin");

    std::cout << "PASSED\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Scalar Types ===\n\n";

    test_round_trip("int", make_test_int, "value");
    test_round_trip("double", make_test_double, "value");
    test_round_trip("string", make_test_string, "message");

    std::cout << "\n=== Fixed-Size Arrays ===\n\n";

    test_round_trip("std::array<double, 3>", make_test_vec3_double, "position");
    test_round_trip("std::array<int, 3>", make_test_vec3_int, "resolution");

    std::cout << "\n=== Dynamic Arrays ===\n\n";

    test_round_trip("vector<double>", make_test_vector_double, "data");
    test_round_trip("vector<double> (empty)", make_test_vector_double_empty, "data");
    test_round_trip("vector<double> (large)", make_test_vector_double_large, "data");
    test_round_trip("vector<string>", make_test_vector_string, "products");
    test_round_trip("vector<string> (empty)", make_test_vector_string_empty, "products");

    std::cout << "\n=== Compound Types ===\n\n";

    test_round_trip("particle_t", make_test_particle, "particle");
    test_round_trip("vector<particle_t>", make_test_vector_particle, "particles");
    test_round_trip("grid_config_t", make_test_grid_config, "grid");
    test_round_trip("simulation_state_t", make_test_simulation_state, "state");
    test_round_trip("nested_strings_t", make_test_nested_strings, "nested");
    test_round_trip("nested_strings_t (empty)", make_test_nested_strings_empty, "nested");

    std::cout << "\n=== Additional Tests ===\n\n";

    test_binary_vs_ascii_size();
    test_binary_file_io();

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
