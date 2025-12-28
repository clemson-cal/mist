#pragma once

// =============================================================================
// Serialize - Standalone C++20 Serialization Library
// =============================================================================
//
// This library provides a concept-based serialization framework with:
// - Archive concepts (ArchiveWriter, ArchiveReader)
// - Self-describing binary format
// - Human-readable ASCII format
// - Automatic serialization for types with fields() method
// - Enum serialization via ADL to_string/from_string
//
// Basic usage:
//   #include <serialize/serialize.hpp>
//
//   struct my_config {
//       int count;
//       double rate;
//       std::string name;
//
//       auto fields() const {
//           return std::make_tuple(
//               serialize::field("count", count),
//               serialize::field("rate", rate),
//               serialize::field("name", name)
//           );
//       }
//       auto fields() {
//           return std::make_tuple(
//               serialize::field("count", count),
//               serialize::field("rate", rate),
//               serialize::field("name", name)
//           );
//       }
//   };
//
//   // Write
//   std::ofstream file("config.bin", std::ios::binary);
//   serialize::binary_writer writer(file);
//   serialize::serialize(writer, "config", my_config{42, 3.14, "hello"});
//
//   // Read
//   std::ifstream file("config.bin", std::ios::binary);
//   serialize::binary_reader reader(file);
//   my_config cfg;
//   serialize::deserialize(reader, "config", cfg);
//
// =============================================================================

#include "core.hpp"
#include "binary_writer.hpp"
#include "binary_reader.hpp"
#include "ascii_writer.hpp"
#include "ascii_reader.hpp"
#include "parallel.hpp"
#include "parallel_io.hpp"
