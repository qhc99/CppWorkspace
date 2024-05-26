cmake_minimum_required(VERSION 3.20)

add_library(central_lib
        STATIC
        DisjointSet.cpp
        utils.cpp
        numerics.cpp)
target_include_directories(central_lib PUBLIC public)

add_executable(
        lib_main
        main.cpp)
target_compile_options(lib_main PUBLIC $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:-g>)
target_link_options(lib_main PUBLIC $<$<AND:$<CXX_COMPILER_ID:Clang>,$<CONFIG:Debug>>:-g>)
target_link_libraries(lib_main PUBLIC central_lib)
