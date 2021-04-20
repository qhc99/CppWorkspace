cmake_minimum_required(VERSION 3.16)

project(utils)
set(CMAKE_CXX_STANDARD 17)

add_library(pack1
        STATIC
        utils.cpp
        )
add_library(org::nathan::utils ALIAS pack1)
target_include_directories(pack1 PUBLIC include/utils)

add_library(pack2
        STATIC
        miscellaneous.cpp
        )
add_library(org::nathan::utils::miscellaneous ALIAS pack2)
target_include_directories(pack2 PUBLIC include/miscellaneous)