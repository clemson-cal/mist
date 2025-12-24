add_executable(test_serialize test_serialize.cpp)
target_link_libraries(test_serialize PRIVATE mist)
target_compile_options(test_serialize PRIVATE -O2)
add_test(NAME serialize COMMAND test_serialize)

add_executable(test_ndarray test_ndarray.cpp)
target_link_libraries(test_ndarray PRIVATE mist)
target_compile_options(test_ndarray PRIVATE -O2)
add_test(NAME ndarray COMMAND test_ndarray)

add_executable(test_comm test_comm.cpp)
target_link_libraries(test_comm PRIVATE mist)
target_compile_options(test_comm PRIVATE -O2)
add_test(NAME comm COMMAND test_comm)

add_executable(write_test_binaries write_test_binaries.cpp)
target_link_libraries(write_test_binaries PRIVATE mist)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME mist_archive
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_mist_archive.py
            ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(mist_archive PROPERTIES
        FIXTURES_REQUIRED test_binaries)
    add_test(NAME write_test_binaries
        COMMAND write_test_binaries ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(write_test_binaries PROPERTIES
        FIXTURES_SETUP test_binaries)
endif()
