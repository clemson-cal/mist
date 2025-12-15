add_executable(srhd1d srhd1d.cpp)
target_link_libraries(srhd1d PRIVATE mist)
mist_add_header_dependencies(srhd1d)

if(READLINE_LIBRARY AND READLINE_INCLUDE_DIR)
    target_include_directories(srhd1d PRIVATE ${READLINE_INCLUDE_DIR})
    target_link_libraries(srhd1d PRIVATE ${READLINE_LIBRARY})
else()
    message(WARNING "readline not found - srhd1d will build without readline support")
endif()

target_compile_options(srhd1d PRIVATE -O3)
