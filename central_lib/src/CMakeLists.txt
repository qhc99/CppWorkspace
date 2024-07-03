cmake_minimum_required(VERSION 3.20)

add_library(central_lib
        STATIC
        DisjointSet.cpp
        utils.cpp
        numeric_utils.cpp)
target_include_directories(central_lib PUBLIC public)

find_package(fmt CONFIG REQUIRED)
add_executable(
        lib_main
        main.cpp)
target_link_libraries(lib_main PRIVATE central_lib fmt::fmt)
