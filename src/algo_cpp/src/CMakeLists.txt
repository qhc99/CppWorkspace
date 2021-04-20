cmake_minimum_required(VERSION 3.16)

project(algo_cpp)
set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME}
        SHARED
        DisjointSet.cpp
        )
add_library(org::nathan::algo_cpp ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC include)
