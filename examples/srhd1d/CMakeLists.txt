add_executable(srhd1d srhd1d.cpp)
target_link_libraries(srhd1d PRIVATE mist_driver)
mist_add_header_dependencies(srhd1d)
target_compile_options(srhd1d PRIVATE -O3)
