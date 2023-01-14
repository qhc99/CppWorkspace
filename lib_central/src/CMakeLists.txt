cmake_minimum_required(VERSION 3.20)

add_library(lib_central
        STATIC
        DisjointSet.cpp
        utils.cpp
        numerics.cpp
        )
add_library(dev::qhc::lib_central ALIAS lib_central)
target_include_directories(lib_central PUBLIC public)

add_executable(
        lib_main
        main.cpp)
target_link_libraries(lib_main PUBLIC dev::qhc::lib_central)

