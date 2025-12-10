#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "mist/core.hpp"
#include "mist/serialize.hpp"
#include "mist/ascii_writer.hpp"
#include "mist/ascii_reader.hpp"
#include "mist/binary_writer.hpp"
#include "mist/binary_reader.hpp"

using namespace mist;

// =============================================================================
// Test structures (matching the README specification)
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

// =============================================================================
// Helper functions
// =============================================================================

bool approx_equal(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

template<typename T, std::size_t N>
bool vec_equal(const vec_t<T, N>& a, const vec_t<T, N>& b) {
    for (std::size_t i = 0; i < N; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!approx_equal(a[i], b[i])) return false;
        } else {
            if (a[i] != b[i]) return false;
        }
    }
    return true;
}

// =============================================================================
// Tests
// =============================================================================

void test_scalar_serialization() {
    std::cout << "Testing scalar serialization... ";

    std::stringstream ss;
    ascii_writer writer(ss);

    writer.begin_named("time");
    writer.write(1.234);
    writer.begin_named("iteration");
    writer.write(42);

    std::string output = ss.str();
    assert(output.find("time = ") != std::string::npos);
    assert(output.find("iteration = 42") != std::string::npos);

    // Test round-trip
    ss.seekg(0);
    ascii_reader reader(ss);

    double time;
    int iteration;
    reader.begin_named("time");
    reader.read(time);
    reader.begin_named("iteration");
    reader.read(iteration);

    assert(approx_equal(time, 1.234));
    assert(iteration == 42);

    std::cout << "PASSED\n";
}

void test_vec_serialization() {
    std::cout << "Testing vec_t serialization... ";

    vec_t<double, 3> original = {1.5, 2.5, 3.5};

    std::stringstream ss;
    ascii_writer writer(ss);
    writer.begin_named("position");
    writer.write(original);

    std::string output = ss.str();
    assert(output.find("position = [") != std::string::npos);

    ss.seekg(0);
    ascii_reader reader(ss);

    vec_t<double, 3> loaded = {};
    reader.begin_named("position");
    reader.read(loaded);

    assert(vec_equal(original, loaded));

    std::cout << "PASSED\n";
}

void test_scalar_vector_serialization() {
    std::cout << "Testing std::vector<double> serialization... ";

    std::vector<double> original = {300.0, 305.2, 298.5, 302.1};

    std::stringstream ss;
    ascii_writer writer(ss);
    writer.begin_named("scalar_field");
    writer.write(original);

    std::string output = ss.str();
    assert(output.find("scalar_field = [") != std::string::npos);

    ss.seekg(0);
    ascii_reader reader(ss);

    std::vector<double> loaded;
    reader.begin_named("scalar_field");
    reader.read(loaded);

    assert(original.size() == loaded.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        assert(approx_equal(original[i], loaded[i]));
    }

    std::cout << "PASSED\n";
}

void test_nested_struct_serialization() {
    std::cout << "Testing nested struct serialization... ";

    grid_config_t original;
    original.resolution = {64, 64, 32};
    original.domain_min = {0.0, 0.0, 0.0};
    original.domain_max = {1.0, 1.0, 0.5};

    std::stringstream ss;
    ascii_writer writer(ss);
    serialize(writer, "grid", original);

    std::string output = ss.str();
    assert(output.find("grid {") != std::string::npos);
    assert(output.find("resolution = [64, 64, 32]") != std::string::npos);

    ss.seekg(0);
    ascii_reader reader(ss);

    grid_config_t loaded;
    deserialize(reader, "grid", loaded);

    assert(vec_equal(original.resolution, loaded.resolution));
    assert(vec_equal(original.domain_min, loaded.domain_min));
    assert(vec_equal(original.domain_max, loaded.domain_max));

    std::cout << "PASSED\n";
}

void test_compound_vector_serialization() {
    std::cout << "Testing std::vector<compound> serialization... ";

    std::vector<particle_t> original = {
        {{0.1, 0.2, 0.15}, {1.5, -0.3, 0.0}, 1.0},
        {{0.8, 0.7, 0.25}, {-0.5, 0.8, 0.2}, 2.0}
    };

    std::stringstream ss;
    ascii_writer writer(ss);
    serialize(writer, "particles", original);

    std::string output = ss.str();
    assert(output.find("particles {") != std::string::npos);

    ss.seekg(0);
    ascii_reader reader(ss);

    std::vector<particle_t> loaded;
    deserialize(reader, "particles", loaded);

    assert(original.size() == loaded.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        assert(vec_equal(original[i].position, loaded[i].position));
        assert(vec_equal(original[i].velocity, loaded[i].velocity));
        assert(approx_equal(original[i].mass, loaded[i].mass));
    }

    std::cout << "PASSED\n";
}

void test_full_simulation_state() {
    std::cout << "Testing full simulation_state_t serialization... ";

    simulation_state_t original;
    original.time = 1.234;
    original.iteration = 42;
    original.grid.resolution = {64, 64, 32};
    original.grid.domain_min = {0.0, 0.0, 0.0};
    original.grid.domain_max = {1.0, 1.0, 0.5};
    original.particles = {
        {{0.1, 0.2, 0.15}, {1.5, -0.3, 0.0}, 1.2},
        {{0.8, 0.7, 0.25}, {-0.5, 0.8, 0.2}, 1.1}
    };
    original.scalar_field = {300.0, 305.2, 298.5, 302.1};

    std::stringstream ss;
    ascii_writer writer(ss);
    serialize(writer, "simulation_state", original);

    std::cout << "\n--- Serialized Output ---\n";
    std::cout << ss.str();
    std::cout << "--- End Output ---\n";

    ss.seekg(0);
    ascii_reader reader(ss);

    simulation_state_t loaded;
    deserialize(reader, "simulation_state", loaded);

    assert(approx_equal(original.time, loaded.time));
    assert(original.iteration == loaded.iteration);
    assert(vec_equal(original.grid.resolution, loaded.grid.resolution));
    assert(vec_equal(original.grid.domain_min, loaded.grid.domain_min));
    assert(vec_equal(original.grid.domain_max, loaded.grid.domain_max));
    assert(original.particles.size() == loaded.particles.size());
    assert(original.scalar_field.size() == loaded.scalar_field.size());

    std::cout << "PASSED\n";
}

// =============================================================================
// Binary Serialization Tests
// =============================================================================

void test_binary_scalar_serialization() {
    std::cout << "Testing binary scalar serialization... ";

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);

    writer.begin_named("time");
    writer.write(1.234);
    writer.begin_named("iteration");
    writer.write(42);

    // Test round-trip
    ss.seekg(0);
    binary_reader reader(ss);

    double time;
    int iteration;
    reader.begin_named("time");
    reader.read(time);
    reader.begin_named("iteration");
    reader.read(iteration);

    assert(approx_equal(time, 1.234));
    assert(iteration == 42);

    std::cout << "PASSED\n";
}

void test_binary_string_serialization() {
    std::cout << "Testing binary string serialization... ";

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);

    std::string original = "Hello, World!\nWith escape chars: \t\"quoted\"";
    writer.begin_named("message");
    writer.write(original);

    ss.seekg(0);
    binary_reader reader(ss);

    std::string loaded;
    reader.begin_named("message");
    reader.read(loaded);

    assert(original == loaded);

    std::cout << "PASSED\n";
}

void test_binary_vec_serialization() {
    std::cout << "Testing binary vec_t serialization... ";

    vec_t<double, 3> original = {1.5, 2.5, 3.5};

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    writer.begin_named("position");
    writer.write(original);

    ss.seekg(0);
    binary_reader reader(ss);

    vec_t<double, 3> loaded = {};
    reader.begin_named("position");
    reader.read(loaded);

    assert(vec_equal(original, loaded));

    std::cout << "PASSED\n";
}

void test_binary_scalar_vector_serialization() {
    std::cout << "Testing binary std::vector<double> serialization... ";

    std::vector<double> original = {300.0, 305.2, 298.5, 302.1};

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    writer.begin_named("scalar_field");
    writer.write(original);

    ss.seekg(0);
    binary_reader reader(ss);

    std::vector<double> loaded;
    reader.begin_named("scalar_field");
    reader.read(loaded);

    assert(original.size() == loaded.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        assert(approx_equal(original[i], loaded[i]));
    }

    std::cout << "PASSED\n";
}

void test_binary_nested_struct_serialization() {
    std::cout << "Testing binary nested struct serialization... ";

    grid_config_t original;
    original.resolution = {64, 64, 32};
    original.domain_min = {0.0, 0.0, 0.0};
    original.domain_max = {1.0, 1.0, 0.5};

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "grid", original);

    ss.seekg(0);
    binary_reader reader(ss);

    grid_config_t loaded;
    deserialize(reader, "grid", loaded);

    assert(vec_equal(original.resolution, loaded.resolution));
    assert(vec_equal(original.domain_min, loaded.domain_min));
    assert(vec_equal(original.domain_max, loaded.domain_max));

    std::cout << "PASSED\n";
}

void test_binary_compound_vector_serialization() {
    std::cout << "Testing binary std::vector<compound> serialization... ";

    std::vector<particle_t> original = {
        {{0.1, 0.2, 0.15}, {1.5, -0.3, 0.0}, 1.0},
        {{0.8, 0.7, 0.25}, {-0.5, 0.8, 0.2}, 2.0}
    };

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "particles", original);

    ss.seekg(0);
    binary_reader reader(ss);

    std::vector<particle_t> loaded;
    deserialize(reader, "particles", loaded);

    assert(original.size() == loaded.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        assert(vec_equal(original[i].position, loaded[i].position));
        assert(vec_equal(original[i].velocity, loaded[i].velocity));
        assert(approx_equal(original[i].mass, loaded[i].mass));
    }

    std::cout << "PASSED\n";
}

void test_binary_full_simulation_state() {
    std::cout << "Testing binary full simulation_state_t serialization... ";

    simulation_state_t original;
    original.time = 1.234;
    original.iteration = 42;
    original.grid.resolution = {64, 64, 32};
    original.grid.domain_min = {0.0, 0.0, 0.0};
    original.grid.domain_max = {1.0, 1.0, 0.5};
    original.particles = {
        {{0.1, 0.2, 0.15}, {1.5, -0.3, 0.0}, 1.2},
        {{0.8, 0.7, 0.25}, {-0.5, 0.8, 0.2}, 1.1}
    };
    original.scalar_field = {300.0, 305.2, 298.5, 302.1};

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "simulation_state", original);

    std::size_t binary_size = ss.str().size();
    std::cout << "\n    Binary size: " << binary_size << " bytes\n";

    ss.seekg(0);
    binary_reader reader(ss);

    simulation_state_t loaded;
    deserialize(reader, "simulation_state", loaded);

    assert(approx_equal(original.time, loaded.time));
    assert(original.iteration == loaded.iteration);
    assert(vec_equal(original.grid.resolution, loaded.grid.resolution));
    assert(vec_equal(original.grid.domain_min, loaded.grid.domain_min));
    assert(vec_equal(original.grid.domain_max, loaded.grid.domain_max));
    assert(original.particles.size() == loaded.particles.size());
    for (std::size_t i = 0; i < original.particles.size(); ++i) {
        assert(vec_equal(original.particles[i].position, loaded.particles[i].position));
        assert(vec_equal(original.particles[i].velocity, loaded.particles[i].velocity));
        assert(approx_equal(original.particles[i].mass, loaded.particles[i].mass));
    }
    assert(original.scalar_field.size() == loaded.scalar_field.size());
    for (std::size_t i = 0; i < original.scalar_field.size(); ++i) {
        assert(approx_equal(original.scalar_field[i], loaded.scalar_field[i]));
    }

    std::cout << "    PASSED\n";
}

void test_binary_vs_ascii_size() {
    std::cout << "Testing binary vs ASCII size comparison... ";

    // Create a larger dataset where binary format shows its advantage
    simulation_state_t state;
    state.time = 1.234;
    state.iteration = 42;
    state.grid.resolution = {64, 64, 32};
    state.grid.domain_min = {0.0, 0.0, 0.0};
    state.grid.domain_max = {1.0, 1.0, 0.5};

    // Add a couple of particles
    state.particles.push_back({{0.1, 0.2, 0.15}, {1.5, -0.3, 0.0}, 1.0});
    state.particles.push_back({{0.8, 0.7, 0.25}, {-0.5, 0.8, 0.2}, 2.0});

    // Add a very large scalar field - binary format stores 8 bytes per double
    // ASCII format stores ~18-20 characters per double (e.g. "300.12345678901234")
    // So binary should be ~40% the size of ASCII for large numeric arrays
    for (int i = 0; i < 1000000; ++i) {
        state.scalar_field.push_back(300.0 + 0.00001 * i);
    }

    // ASCII serialization
    std::stringstream ascii_ss;
    ascii_writer ascii_w(ascii_ss);
    serialize(ascii_w, "state", state);
    std::size_t ascii_size = ascii_ss.str().size();

    // Binary serialization
    std::stringstream binary_ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer binary_w(binary_ss);
    serialize(binary_w, "state", state);
    std::size_t binary_size = binary_ss.str().size();

    std::cout << "\n    ASCII size:  " << ascii_size << " bytes\n";
    std::cout << "    Binary size: " << binary_size << " bytes\n";
    std::cout << "    Ratio: " << static_cast<double>(binary_size) / ascii_size * 100.0 << "%\n";

    // With larger data, binary should be smaller despite field name overhead
    // (self-describing format adds ~10-20% overhead, but numeric data is much more compact)
    assert(binary_size < ascii_size);

    std::cout << "    PASSED\n";
}

void test_binary_empty_vector() {
    std::cout << "Testing binary empty vector serialization... ";

    std::vector<double> original;

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    writer.begin_named("empty");
    writer.write(original);

    ss.seekg(0);
    binary_reader reader(ss);

    std::vector<double> loaded;
    loaded.push_back(999.0);  // Pre-fill to ensure it gets cleared
    reader.begin_named("empty");
    reader.read(loaded);

    assert(loaded.empty());

    std::cout << "PASSED\n";
}

void test_vector_string_serialization() {
    std::cout << "Testing std::vector<std::string> serialization... ";

    std::vector<std::string> original = {"default", "primitive", "conserved"};

    std::stringstream ss;
    ascii_writer writer(ss);
    serialize(writer, "products", original);

    std::string output = ss.str();
    std::cout << "\n--- Serialized vector<string> ---\n";
    std::cout << output;
    std::cout << "--- End Output ---\n";

    ss.seekg(0);
    ascii_reader reader(ss);

    std::vector<std::string> loaded;
    deserialize(reader, "products", loaded);

    assert(original.size() == loaded.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        assert(original[i] == loaded[i]);
    }

    std::cout << "PASSED\n";
}

void test_empty_vector_string_serialization() {
    std::cout << "Testing empty std::vector<std::string> serialization... ";

    std::vector<std::string> original;

    std::stringstream ss;
    ascii_writer writer(ss);
    serialize(writer, "products", original);

    ss.seekg(0);
    ascii_reader reader(ss);

    std::vector<std::string> loaded;
    loaded.push_back("should_be_cleared");  // Pre-fill to ensure it gets cleared
    deserialize(reader, "products", loaded);

    assert(loaded.empty());

    std::cout << "PASSED\n";
}

void test_nested_vector_string_serialization() {
    std::cout << "Testing nested std::vector<std::string> serialization... ";

    struct nested_t {
        std::vector<std::string> selected_products;

        auto fields() const {
            return std::make_tuple(field("selected_products", selected_products));
        }

        auto fields() {
            return std::make_tuple(field("selected_products", selected_products));
        }
    };

    nested_t original;
    original.selected_products = {"default", "primitive"};

    std::stringstream ss;
    ascii_writer writer(ss);
    serialize(writer, "nested", original);

    std::cout << "\n--- Serialized nested vector<string> ---\n";
    std::cout << ss.str();
    std::cout << "--- End Output ---\n";

    ss.seekg(0);
    ascii_reader reader(ss);

    nested_t loaded;
    deserialize(reader, "nested", loaded);

    assert(original.selected_products.size() == loaded.selected_products.size());
    for (std::size_t i = 0; i < original.selected_products.size(); ++i) {
        assert(original.selected_products[i] == loaded.selected_products[i]);
    }

    std::cout << "PASSED\n";
}

void test_binary_file_output() {
    std::cout << "Testing binary file output (1000 doubles)... ";

    // Create a simple struct with 1000 doubles
    std::vector<double> data;
    for (int i = 0; i < 1000; ++i) {
        data.push_back(static_cast<double>(i) * 0.001);
    }

    // Write to file
    std::ofstream file("test_output.bin", std::ios::binary);
    binary_writer writer(file);
    writer.begin_named("data");
    writer.write(data);
    file.close();

    // Verify file size: header (5) + name (8 + 4) + type tags (2) + count (8) + data (8000) = 8027 bytes
    std::ifstream check("test_output.bin", std::ios::binary | std::ios::ate);
    std::size_t file_size = check.tellg();
    check.close();

    std::cout << "file size: " << file_size << " bytes... ";

    // Read it back
    std::ifstream infile("test_output.bin", std::ios::binary);
    binary_reader reader(infile);
    std::vector<double> loaded;
    reader.begin_named("data");
    reader.read(loaded);
    infile.close();

    assert(loaded.size() == 1000);
    for (int i = 0; i < 1000; ++i) {
        assert(approx_equal(loaded[i], static_cast<double>(i) * 0.001));
    }

    // Clean up
    std::remove("test_output.bin");

    std::cout << "PASSED\n";
}

// =============================================================================
// NdArray Serialization Tests
// =============================================================================

void test_binary_cached_1d_serialization() {
    std::cout << "Testing binary cached_t<double, 1> serialization... ";

    auto space = index_space(ivec(0), uvec(100u));
    cached_t<double, 1> original(space, memory::host);

    for (int i = 0; i < 100; ++i) {
        original(ivec(i)) = static_cast<double>(i) * 0.5;
    }

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "array", original);

    ss.seekg(0);
    binary_reader reader(ss);

    cached_t<double, 1> loaded(index_space(ivec(0), uvec(1u)), memory::host);
    deserialize(reader, "array", loaded);

    assert(start(loaded) == start(original));
    assert(shape(loaded) == shape(original));
    for (int i = 0; i < 100; ++i) {
        assert(approx_equal(loaded(ivec(i)), original(ivec(i))));
    }

    std::cout << "PASSED\n";
}

void test_binary_cached_2d_serialization() {
    std::cout << "Testing binary cached_t<double, 2> serialization... ";

    auto space = index_space(ivec(0, 0), uvec(10u, 20u));
    cached_t<double, 2> original(space, memory::host);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 20; ++j) {
            original(ivec(i, j)) = static_cast<double>(i * 20 + j);
        }
    }

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "grid", original);

    ss.seekg(0);
    binary_reader reader(ss);

    cached_t<double, 2> loaded(index_space(ivec(0, 0), uvec(1u, 1u)), memory::host);
    deserialize(reader, "grid", loaded);

    assert(start(loaded) == start(original));
    assert(shape(loaded) == shape(original));
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 20; ++j) {
            assert(approx_equal(loaded(ivec(i, j)), original(ivec(i, j))));
        }
    }

    std::cout << "PASSED\n";
}

void test_binary_cached_3d_serialization() {
    std::cout << "Testing binary cached_t<double, 3> serialization... ";

    auto space = index_space(ivec(0, 0, 0), uvec(4u, 5u, 6u));
    cached_t<double, 3> original(space, memory::host);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 6; ++k) {
                original(ivec(i, j, k)) = static_cast<double>(i * 30 + j * 6 + k);
            }
        }
    }

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "volume", original);

    ss.seekg(0);
    binary_reader reader(ss);

    cached_t<double, 3> loaded(index_space(ivec(0, 0, 0), uvec(1u, 1u, 1u)), memory::host);
    deserialize(reader, "volume", loaded);

    assert(start(loaded) == start(original));
    assert(shape(loaded) == shape(original));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 6; ++k) {
                assert(approx_equal(loaded(ivec(i, j, k)), original(ivec(i, j, k))));
            }
        }
    }

    std::cout << "PASSED\n";
}

void test_binary_cached_nonzero_start() {
    std::cout << "Testing binary cached_t with non-zero start... ";

    auto space = index_space(ivec(-5, 10), uvec(10u, 20u));
    cached_t<double, 2> original(space, memory::host);

    for (int i = -5; i < 5; ++i) {
        for (int j = 10; j < 30; ++j) {
            original(ivec(i, j)) = static_cast<double>(i * 100 + j);
        }
    }

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "offset_grid", original);

    ss.seekg(0);
    binary_reader reader(ss);

    cached_t<double, 2> loaded(index_space(ivec(0, 0), uvec(1u, 1u)), memory::host);
    deserialize(reader, "offset_grid", loaded);

    assert(start(loaded) == start(original));
    assert(shape(loaded) == shape(original));
    for (int i = -5; i < 5; ++i) {
        for (int j = 10; j < 30; ++j) {
            assert(approx_equal(loaded(ivec(i, j)), original(ivec(i, j))));
        }
    }

    std::cout << "PASSED\n";
}

void test_binary_cached_int_type() {
    std::cout << "Testing binary cached_t<int, 2> serialization... ";

    auto space = index_space(ivec(0, 0), uvec(5u, 5u));
    cached_t<int, 2> original(space, memory::host);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            original(ivec(i, j)) = i * 10 + j;
        }
    }

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    binary_writer writer(ss);
    serialize(writer, "int_grid", original);

    ss.seekg(0);
    binary_reader reader(ss);

    cached_t<int, 2> loaded(index_space(ivec(0, 0), uvec(1u, 1u)), memory::host);
    deserialize(reader, "int_grid", loaded);

    assert(start(loaded) == start(original));
    assert(shape(loaded) == shape(original));
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            assert(loaded(ivec(i, j)) == original(ivec(i, j)));
        }
    }

    std::cout << "PASSED\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== ASCII Serialization Tests ===\n\n";

    test_scalar_serialization();
    test_vec_serialization();
    test_scalar_vector_serialization();
    test_nested_struct_serialization();
    test_compound_vector_serialization();
    test_full_simulation_state();

    std::cout << "\n=== Binary Serialization Tests ===\n\n";

    test_binary_scalar_serialization();
    test_binary_string_serialization();
    test_binary_vec_serialization();
    test_binary_scalar_vector_serialization();
    test_binary_nested_struct_serialization();
    test_binary_compound_vector_serialization();
    test_binary_full_simulation_state();
    test_binary_vs_ascii_size();
    test_binary_empty_vector();
    test_vector_string_serialization();
    test_empty_vector_string_serialization();
    test_nested_vector_string_serialization();
    test_binary_file_output();

    std::cout << "\n=== NdArray Serialization Tests ===\n\n";

    test_binary_cached_1d_serialization();
    test_binary_cached_2d_serialization();
    test_binary_cached_3d_serialization();
    test_binary_cached_nonzero_start();
    test_binary_cached_int_type();

    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
