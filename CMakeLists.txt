cmake_minimum_required(VERSION 3.20)

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Use: cmake -B build")
endif()

project(mist VERSION 0.1.0 LANGUAGES CXX)

# Get git information
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MIST_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MIST_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_STATUS_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GIT_STATUS_OUTPUT)
        set(MIST_GIT_DIRTY true)
    else()
        set(MIST_GIT_DIRTY false)
    endif()
else()
    set(MIST_GIT_COMMIT "unknown")
    set(MIST_GIT_BRANCH "unknown")
    set(MIST_GIT_DIRTY false)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional MPI support
option(MIST_WITH_MPI "Enable MPI support for distributed communication" OFF)
if(MIST_WITH_MPI)
    find_package(MPI REQUIRED)
    set(MIST_WITH_MPI_01 1)
else()
    set(MIST_WITH_MPI_01 0)
endif()

# Generate build_info.hpp from template
configure_file(
    ${CMAKE_SOURCE_DIR}/include/mist/build_info.hpp.in
    ${CMAKE_BINARY_DIR}/include/mist/build_info.hpp
    @ONLY
)

# Header-only core library
add_library(mist INTERFACE)
target_include_directories(mist INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(MIST_WITH_MPI)
    target_link_libraries(mist INTERFACE MPI::MPI_CXX)
    target_compile_definitions(mist INTERFACE MIST_WITH_MPI)
endif()

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
