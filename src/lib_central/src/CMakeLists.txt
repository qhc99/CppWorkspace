cmake_minimum_required(VERSION 3.16)

project(lib_central)

add_library(${PROJECT_NAME}
        STATIC
        DisjointSet.cpp
        include/lib_central/numerics.h
        include/lib_central/rank_search.hpp
        include/lib_central/DisjointSet.h
        utils.cpp
        numerics.cpp)
add_library(dev::qhc::lib_central ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC include)
