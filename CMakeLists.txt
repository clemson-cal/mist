cmake_minimum_required(VERSION 3.20)
project(mist VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only library
add_library(mist INTERFACE)
target_include_directories(mist INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find readline (required for interactive examples)
find_library(READLINE_LIBRARY readline)
find_path(READLINE_INCLUDE_DIR readline/readline.h
    HINTS /opt/homebrew/opt/readline/include /usr/local/include
)

# Build examples and tests
option(MIST_BUILD_EXAMPLES "Build example programs" ON)
option(MIST_BUILD_TESTS "Build tests" ON)

if(MIST_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MIST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
