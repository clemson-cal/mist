add_executable(test_archive test_archive.cpp)
target_link_libraries(test_archive PRIVATE mist)
target_compile_options(test_archive PRIVATE -O2)
add_test(NAME archive COMMAND test_archive)

add_executable(test_ndarray test_ndarray.cpp)
target_link_libraries(test_ndarray PRIVATE mist)
target_compile_options(test_ndarray PRIVATE -O2)
add_test(NAME ndarray COMMAND test_ndarray)

add_executable(test_comm test_comm.cpp)
target_link_libraries(test_comm PRIVATE mist)
target_compile_options(test_comm PRIVATE -O2)
add_test(NAME comm COMMAND test_comm)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE mist)
target_compile_options(test_tensor PRIVATE -O2)
add_test(NAME tensor COMMAND test_tensor)

if(MIST_WITH_MPI)
    add_executable(test_comm_mpi test_comm_mpi.cpp)
    target_link_libraries(test_comm_mpi PRIVATE mist)
    target_compile_options(test_comm_mpi PRIVATE -O2)
    add_test(NAME comm_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_comm_mpi>)
endif()

add_executable(write_test_binaries write_test_binaries.cpp)
target_link_libraries(write_test_binaries PRIVATE mist)

# Prefer venv Python if it exists, otherwise use system Python
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
else()
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    endif()
endif()

if(PYTHON_EXECUTABLE)
    add_test(NAME mist_archive
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_mist_archive.py
            ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(mist_archive PROPERTIES
        FIXTURES_REQUIRED test_binaries)
    add_test(NAME write_test_binaries
        COMMAND write_test_binaries ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(write_test_binaries PROPERTIES
        FIXTURES_SETUP test_binaries)
endif()
