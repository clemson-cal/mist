add_executable(test_serialize test_serialize.cpp)
target_link_libraries(test_serialize PRIVATE mist)
target_compile_options(test_serialize PRIVATE -O2)
add_test(NAME serialize COMMAND test_serialize)

add_executable(test_ndarray test_ndarray.cpp)
target_link_libraries(test_ndarray PRIVATE mist)
target_compile_options(test_ndarray PRIVATE -O2)
add_test(NAME ndarray COMMAND test_ndarray)
