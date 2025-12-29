#pragma once

// ============================================================================
// Archive - C++20 Serialization Library
// ============================================================================
//
// A concept-based serialization library with:
// - Key-based access for forward/backward compatibility
// - Self-describing binary format
// - Human-readable ASCII format
// - Automatic serialization for types with ADL fields() function
// - Enum serialization via ADL to_string/from_string
//
// Basic usage:
//
//   #include "archive.hpp"
//
//   struct my_config {
//       int count = 0;
//       double rate = 1.0;
//       std::string name;
//   };
//
//   // ADL free function for serialization (required)
//   auto fields(const my_config& c) {
//       return std::make_tuple(
//           archive::field("count", c.count),
//           archive::field("rate", c.rate),
//           archive::field("name", c.name)
//       );
//   }
//   auto fields(my_config& c) {
//       return std::make_tuple(
//           archive::field("count", c.count),
//           archive::field("rate", c.rate),
//           archive::field("name", c.name)
//       );
//   }
//
//   // Write (binary)
//   std::ofstream file("config.bin", std::ios::binary);
//   archive::binary_sink sink(file);
//   archive::write(sink, "config", my_config{42, 3.14, "hello"});
//
//   // Read (binary)
//   std::ifstream file("config.bin", std::ios::binary);
//   archive::binary_source source(file);
//   my_config cfg;
//   archive::read(source, "config", cfg);
//
//   // Write (ASCII)
//   std::ofstream file("config.dat");
//   archive::ascii_sink sink(file);
//   archive::write(sink, "config", my_config{42, 3.14, "hello"});
//
//   // Read (ASCII)
//   std::ifstream file("config.dat");
//   archive::ascii_source source(file);
//   my_config cfg;
//   archive::read(source, "config", cfg);
//
// ============================================================================

#include "checkpoint.hpp"
#include "format.hpp"
#include "protocol.hpp"
#include "sink.hpp"
#include "source.hpp"
