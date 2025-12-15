add_executable(config-reader config-reader.cpp)
target_link_libraries(config-reader PRIVATE mist)
mist_add_header_dependencies(config-reader)
target_compile_options(config-reader PRIVATE -O2)
