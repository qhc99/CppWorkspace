cmake_minimum_required(VERSION 3.16)

project(lib_central)

add_library(${PROJECT_NAME}
        STATIC
        DisjointSet.cpp
        )
add_library(org::qhc::lib_central ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC include)
