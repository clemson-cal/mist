#include <fstream>
#include <string>
#include <vector>
#include "mist/core.hpp"
#include "mist/archive.hpp"

using namespace mist;

struct config_t {
    int count;
    std::vector<std::string> names;
};

inline auto fields(const config_t& c) {
    return std::make_tuple(
        field("count", c.count),
        field("names", c.names)
    );
}

int main(int argc, char* argv[]) {
    auto dir = std::string{argc > 1 ? argv[1] : "."};

    // vector<string>
    {
        auto file = std::ofstream(dir + "/test_vector_string.bin", std::ios::binary);
        auto sink = binary_sink(file);
        write(sink, "products", std::vector<std::string>{"default", "primitive", "conserved"});
    }

    // empty vector<string>
    {
        auto file = std::ofstream(dir + "/test_vector_string_empty.bin", std::ios::binary);
        auto sink = binary_sink(file);
        write(sink, "products", std::vector<std::string>{});
    }

    // nested struct with vector<string>
    {
        auto file = std::ofstream(dir + "/test_nested_vector_string.bin", std::ios::binary);
        auto sink = binary_sink(file);
        write(sink, "config", config_t{3, {"alpha", "beta", "gamma"}});
    }

    // mixed types
    {
        auto file = std::ofstream(dir + "/test_mixed_types.bin", std::ios::binary);
        auto sink = binary_sink(file);
        write(sink, "name", std::string("test"));
        write(sink, "value", 42);
        write(sink, "pi", 3.14159);
        write(sink, "tags", std::vector<std::string>{"a", "b", "c"});
        write(sink, "data", std::vector<double>{1.0, 2.0, 3.0});
    }

    return 0;
}
