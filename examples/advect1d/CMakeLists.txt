add_executable(advect1d advect1d.cpp)
target_link_libraries(advect1d PRIVATE mist)

if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    target_include_directories(advect1d PRIVATE ${READLINE_INCLUDE_DIR})
    target_link_libraries(advect1d PRIVATE ${READLINE_LIBRARY})
else()
    message(WARNING "readline not found - advect1d will build without readline support")
endif()

target_compile_options(advect1d PRIVATE -O3)
