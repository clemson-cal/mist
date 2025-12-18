add_executable(advect1d advect1d.cpp)
target_link_libraries(advect1d PRIVATE mist_driver)
mist_add_header_dependencies(advect1d)
target_compile_options(advect1d PRIVATE -O3)
