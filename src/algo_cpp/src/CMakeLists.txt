cmake_minimum_required(VERSION 3.16)

project(algo_cpp)
set(CMAKE_CXX_STANDARD 17)

get_filename_component(pack1 include/structures NAME)
add_library(${pack1}
        SHARED
        DisjointSet.cpp
        )
add_library(org::nathan::algo_cpp::structures ALIAS ${pack1})
target_include_directories(${pack1} PUBLIC include/structures)

get_filename_component(pack2 include/miscellaneous NAME)
add_library(${pack2} INTERFACE)
add_library(org::nathan::algo_cpp::miscellaneous ALIAS ${pack2})
target_include_directories(${pack2} INTERFACE include/miscellaneous)