#include <fstream>
#include <string>
#include <vector>
#include "mist/core.hpp"
#include "mist/serialize.hpp"

using namespace mist;

struct config_t {
    int count;
    std::vector<std::string> names;

    auto fields() const {
        return std::make_tuple(
            field("count", count),
            field("names", names)
        );
    }
};

int main(int argc, char* argv[]) {
    auto dir = std::string{argc > 1 ? argv[1] : "."};

    // vector<string>
    {
        auto file = std::ofstream(dir + "/test_vector_string.bin", std::ios::binary);
        auto writer = binary_writer(file);
        mist::serialize(writer, "products", std::vector<std::string>{"default", "primitive", "conserved"});
    }

    // empty vector<string>
    {
        auto file = std::ofstream(dir + "/test_vector_string_empty.bin", std::ios::binary);
        auto writer = binary_writer(file);
        mist::serialize(writer, "products", std::vector<std::string>{});
    }

    // nested struct with vector<string>
    {
        auto file = std::ofstream(dir + "/test_nested_vector_string.bin", std::ios::binary);
        auto writer = binary_writer(file);
        mist::serialize(writer, "config", config_t{3, {"alpha", "beta", "gamma"}});
    }

    // mixed types
    {
        auto file = std::ofstream(dir + "/test_mixed_types.bin", std::ios::binary);
        auto writer = binary_writer(file);
        mist::serialize(writer, "name", std::string("test"));
        mist::serialize(writer, "value", 42);
        mist::serialize(writer, "pi", 3.14159);
        mist::serialize(writer, "tags", std::vector<std::string>{"a", "b", "c"});
        mist::serialize(writer, "data", std::vector<double>{1.0, 2.0, 3.0});
    }

    return 0;
}
