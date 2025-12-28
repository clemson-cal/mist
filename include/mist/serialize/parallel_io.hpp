#pragma once

// =============================================================================
// Parallel IO - filesystem-based parallel serialization
// =============================================================================
//
// Directory structure:
//   path/
//     header.bin    - serialized header data
//     items.txt     - list of item keys (one per line)
//     item_{key}.bin - serialized item data
//
// =============================================================================

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "parallel.hpp"
#include "binary_writer.hpp"
#include "binary_reader.hpp"
#include "core.hpp"

namespace serialize {

namespace fs = std::filesystem;

// =============================================================================
// parallel_write - write state to directory
// =============================================================================

template<ParallelWrite State>
    requires HasItemKey<parallel_item_type<State>>
void parallel_write(const fs::path& path, const State& state) {
    // Create directory
    fs::create_directories(path);

    // Write header
    {
        std::ofstream file(path / "header.bin", std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to open header.bin for writing");
        }
        binary_writer writer(file);
        serialize(writer, header(state));
    }

    // Collect item keys and write items
    std::vector<std::string> keys;
    for (const auto& item : items(state)) {
        auto key = item_key(item);
        keys.push_back(key);

        std::ofstream file(path / ("item_" + key + ".bin"), std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to open item file for writing: " + key);
        }
        binary_writer writer(file);
        serialize(writer, item);
    }

    // Write index file
    {
        std::ofstream file(path / "items.txt");
        if (!file) {
            throw std::runtime_error("failed to open items.txt for writing");
        }
        for (const auto& key : keys) {
            file << key << "\n";
        }
    }
}

// =============================================================================
// parallel_read - read state from directory with predicate filter
// =============================================================================

template<ParallelRead State, typename Predicate>
    requires std::invocable<Predicate, const std::string&>
          && std::convertible_to<std::invoke_result_t<Predicate, const std::string&>, bool>
void parallel_read(const fs::path& path, State& state, Predicate&& wants_item) {
    using item_type = std::ranges::range_value_t<decltype(items(state))>;

    // Read header
    {
        std::ifstream file(path / "header.bin", std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to open header.bin for reading");
        }
        binary_reader reader(file);
        deserialize(reader, header(state));
    }

    // Read index file to get list of available items
    std::vector<std::string> keys;
    {
        std::ifstream file(path / "items.txt");
        if (!file) {
            throw std::runtime_error("failed to open items.txt for reading");
        }
        std::string key;
        while (std::getline(file, key)) {
            if (!key.empty()) {
                keys.push_back(key);
            }
        }
    }

    // Clear existing items
    items(state).clear();

    // Read items that pass the predicate
    for (const auto& key : keys) {
        if (!wants_item(key)) {
            continue;
        }

        std::ifstream file(path / ("item_" + key + ".bin"), std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to open item file for reading: " + key);
        }
        binary_reader reader(file);
        item_type item;
        if (!deserialize(reader, item)) {
            throw std::runtime_error("failed to deserialize item: " + key);
        }
        items(state).push_back(std::move(item));
    }
}

// Convenience overload: read all items (no filtering)
template<ParallelRead State>
void parallel_read(const fs::path& path, State& state) {
    parallel_read(path, state, [](const std::string&) { return true; });
}

// =============================================================================
// Utilities
// =============================================================================

// List available item keys in a parallel IO directory
inline auto list_item_keys(const fs::path& path) -> std::vector<std::string> {
    std::vector<std::string> keys;
    std::ifstream file(path / "items.txt");
    if (!file) {
        throw std::runtime_error("failed to open items.txt for reading");
    }
    std::string key;
    while (std::getline(file, key)) {
        if (!key.empty()) {
            keys.push_back(key);
        }
    }
    return keys;
}

// Read only the header from a parallel IO directory
template<typename Header>
void read_header(const fs::path& path, Header& hdr) {
    std::ifstream file(path / "header.bin", std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open header.bin for reading");
    }
    binary_reader reader(file);
    deserialize(reader, hdr);
}

} // namespace serialize
