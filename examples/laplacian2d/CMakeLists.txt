add_executable(laplacian2d laplacian2d.cpp)
target_link_libraries(laplacian2d PRIVATE mist)
target_compile_options(laplacian2d PRIVATE -O2)
