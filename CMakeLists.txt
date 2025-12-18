cmake_minimum_required(VERSION 3.20)
project(mist VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only core library
add_library(mist INTERFACE)
target_include_directories(mist INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Collect all headers for dependency tracking
file(GLOB_RECURSE MIST_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# Helper function to add mist headers as dependencies to a target
function(mist_add_header_dependencies target)
    target_sources(${target} PRIVATE ${MIST_HEADERS})
    set_source_files_properties(${MIST_HEADERS} PROPERTIES HEADER_FILE_ONLY TRUE)
endfunction()

# Find readline (required for interactive examples and driver)
find_library(READLINE_LIBRARY readline)
find_path(READLINE_INCLUDE_DIR readline/readline.h
    HINTS /opt/homebrew/opt/readline/include /usr/local/include
)

# Driver library (compiled)
add_library(mist_driver STATIC
    src/driver/engine.cpp
    src/driver/repl_session.cpp
    src/driver/socket_session.cpp
)
target_include_directories(mist_driver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(mist_driver PUBLIC mist)
if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    target_link_libraries(mist_driver PUBLIC ${READLINE_LIBRARY})
    target_include_directories(mist_driver PUBLIC ${READLINE_INCLUDE_DIR})
endif()

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
