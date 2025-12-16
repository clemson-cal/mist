#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include "mist/ascii_reader.hpp"
#include "mist/ascii_writer.hpp"
#include "mist/binary_reader.hpp"
#include "mist/binary_writer.hpp"
#include "mist/core.hpp"
#include "mist/serialize.hpp"

using namespace mist;

// =============================================================================
// Test structures
// =============================================================================

struct particle_t {
    vec_t<double, 3> position;
    vec_t<double, 3> velocity;
    double mass;

    auto fields() const {
        return std::make_tuple(
            field("position", position),
            field("velocity", velocity),
            field("mass", mass)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("position", position),
            field("velocity", velocity),
            field("mass", mass)
        );
    }
};

struct grid_config_t {
    vec_t<int, 3> resolution;
    vec_t<double, 3> domain_min;
    vec_t<double, 3> domain_max;

    auto fields() const {
        return std::make_tuple(
            field("resolution", resolution),
            field("domain_min", domain_min),
            field("domain_max", domain_max)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("resolution", resolution),
            field("domain_min", domain_min),
            field("domain_max", domain_max)
        );
    }
};

struct simulation_state_t {
    double time;
    int iteration;
    grid_config_t grid;
    std::vector<particle_t> particles;
    std::vector<double> scalar_field;

    auto fields() const {
        return std::make_tuple(
            field("time", time),
            field("iteration", iteration),
            field("grid", grid),
            field("particles", particles),
            field("scalar_field", scalar_field)
        );
    }

    auto fields() {
        return std::make_tuple(
            field("time", time),
            field("iteration", iteration),
            field("grid", grid),
            field("particles", particles),
            field("scalar_field", scalar_field)
        );
    }
};

struct nested_strings_t {
    std::vector<std::string> items;

    auto fields() const {
        return std::make_tuple(field("items", items));
    }

    auto fields() {
        return std::make_tuple(field("items", items));
    }
};

// =============================================================================
// Equality comparators
// =============================================================================

auto approx_equal(double a, double b, double tol = 1e-10) -> bool {
    return std::abs(a - b) < tol;
}

template<typename T, std::size_t N>
auto equal(const vec_t<T, N>& a, const vec_t<T, N>& b) -> bool {
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

template<std::size_t D>
auto equal(const cached_t<double, D>& a, const cached_t<double, D>& b) -> bool {
    if (start(a) != start(b)) return false;
    if (shape(a) != shape(b)) return false;
    for (auto i = std::size_t{0}; i < size(a); ++i) {
        if (!approx_equal(data(a)[i], data(b)[i])) return false;
    }
    return true;
}

template<std::size_t D>
auto equal(const cached_t<int, D>& a, const cached_t<int, D>& b) -> bool {
    if (start(a) != start(b)) return false;
    if (shape(a) != shape(b)) return false;
    for (auto i = std::size_t{0}; i < size(a); ++i) {
        if (data(a)[i] != data(b)[i]) return false;
    }
    return true;
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

auto make_test_vec3_double() -> vec_t<double, 3> {
    return {1.5, 2.5, 3.5};
}

auto make_test_vec3_int() -> vec_t<int, 3> {
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
    return {{0.1, 0.2, 0.15}, {1.5, -0.3, 0.0}, 1.2};
}

auto make_test_vector_particle() -> std::vector<particle_t> {
    return {
        {{0.1, 0.2, 0.15}, {1.5, -0.3, 0.0}, 1.0},
        {{0.8, 0.7, 0.25}, {-0.5, 0.8, 0.2}, 2.0}
    };
}

auto make_test_grid_config() -> grid_config_t {
    return {{64, 64, 32}, {0.0, 0.0, 0.0}, {1.0, 1.0, 0.5}};
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

auto make_test_cached_1d() -> cached_t<double, 1> {
    auto space = index_space(ivec(0), uvec(100u));
    auto arr = cached_t<double, 1>(space, memory::host);
    for (auto i = 0; i < 100; ++i) {
        arr(ivec(i)) = static_cast<double>(i) * 0.5;
    }
    return arr;
}

auto make_test_cached_2d() -> cached_t<double, 2> {
    auto space = index_space(ivec(0, 0), uvec(10u, 20u));
    auto arr = cached_t<double, 2>(space, memory::host);
    for (auto i = 0; i < 10; ++i) {
        for (auto j = 0; j < 20; ++j) {
            arr(ivec(i, j)) = static_cast<double>(i * 20 + j);
        }
    }
    return arr;
}

auto make_test_cached_3d() -> cached_t<double, 3> {
    auto space = index_space(ivec(0, 0, 0), uvec(4u, 5u, 6u));
    auto arr = cached_t<double, 3>(space, memory::host);
    for (auto i = 0; i < 4; ++i) {
        for (auto j = 0; j < 5; ++j) {
            for (auto k = 0; k < 6; ++k) {
                arr(ivec(i, j, k)) = static_cast<double>(i * 30 + j * 6 + k);
            }
        }
    }
    return arr;
}

auto make_test_cached_2d_offset() -> cached_t<double, 2> {
    auto space = index_space(ivec(-5, 10), uvec(10u, 20u));
    auto arr = cached_t<double, 2>(space, memory::host);
    for (auto i = -5; i < 5; ++i) {
        for (auto j = 10; j < 30; ++j) {
            arr(ivec(i, j)) = static_cast<double>(i * 100 + j);
        }
    }
    return arr;
}

auto make_test_cached_2d_int() -> cached_t<int, 2> {
    auto space = index_space(ivec(0, 0), uvec(5u, 5u));
    auto arr = cached_t<int, 2>(space, memory::host);
    for (auto i = 0; i < 5; ++i) {
        for (auto j = 0; j < 5; ++j) {
            arr(ivec(i, j)) = i * 10 + j;
        }
    }
    return arr;
}

// =============================================================================
// Format tags and round-trip
// =============================================================================

struct ascii_tag {
    static auto make_stream() -> std::stringstream {
        return std::stringstream{};
    }

    static auto make_writer(std::stringstream& ss) -> ascii_writer {
        return ascii_writer(ss);
    }

    static auto make_reader(std::stringstream& ss) -> ascii_reader {
        return ascii_reader(ss);
    }
};

struct binary_tag {
    static auto make_stream() -> std::stringstream {
        return std::stringstream(std::ios::binary | std::ios::in | std::ios::out);
    }

    static auto make_writer(std::stringstream& ss) -> binary_writer {
        return binary_writer(ss);
    }

    static auto make_reader(std::stringstream& ss) -> binary_reader {
        return binary_reader(ss);
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

// Specializations for cached_t which needs non-default construction
template<typename T, std::size_t D>
auto make_default() -> cached_t<T, D> {
    auto dummy_space = index_space(vec_t<int, D>{}, vec_t<unsigned, D>{});
    return cached_t<T, D>(dummy_space, memory::host);
}

template<typename Format, typename T>
auto round_trip(const T& original, const char* name) -> T {
    auto ss = Format::make_stream();
    auto writer = Format::make_writer(ss);
    serialize(writer, name, original);

    ss.seekg(0);
    auto reader = Format::make_reader(ss);
    auto loaded = make_default<T>();
    deserialize(reader, name, loaded);
    return loaded;
}

template<typename T>
auto round_trip_file(const T& original, const char* name) -> T {
    {
        auto file = std::ofstream(binary_file_tag::filename, std::ios::binary);
        auto writer = binary_writer(file);
        serialize(writer, name, original);
    }

    auto loaded = make_default<T>();
    {
        auto file = std::ifstream(binary_file_tag::filename, std::ios::binary);
        auto reader = binary_reader(file);
        deserialize(reader, name, loaded);
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
    auto ascii_w = ascii_writer(ascii_ss);
    serialize(ascii_w, "state", state);
    auto ascii_size = ascii_ss.str().size();

    auto binary_ss = std::stringstream(std::ios::binary | std::ios::in | std::ios::out);
    auto binary_w = binary_writer(binary_ss);
    serialize(binary_w, "state", state);
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
        auto writer = binary_writer(file);
        serialize(writer, "data", original);
    }

    auto loaded = std::vector<double>{};
    {
        auto file = std::ifstream("test_output.bin", std::ios::binary);
        auto reader = binary_reader(file);
        deserialize(reader, "data", loaded);
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

    test_round_trip("vec_t<double, 3>", make_test_vec3_double, "position");
    test_round_trip("vec_t<int, 3>", make_test_vec3_int, "resolution");

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

    std::cout << "\n=== NdArray Types ===\n\n";

    test_round_trip("cached_t<double, 1>", make_test_cached_1d, "array");
    test_round_trip("cached_t<double, 2>", make_test_cached_2d, "grid");
    test_round_trip("cached_t<double, 3>", make_test_cached_3d, "volume");
    test_round_trip("cached_t<double, 2> (offset)", make_test_cached_2d_offset, "offset_grid");
    test_round_trip("cached_t<int, 2>", make_test_cached_2d_int, "int_grid");

    std::cout << "\n=== Additional Tests ===\n\n";

    test_binary_vs_ascii_size();
    test_binary_file_io();

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
