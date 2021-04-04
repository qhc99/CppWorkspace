cmake_minimum_required(VERSION 3.16)

project(algo_cpp_miscellaneous)

add_library(
        ${PROJECT_NAME}
        SHARED
        src/rank_search.cpp
)

add_library(
        org::nathan::algo_cpp::miscellaneous
        ALIAS
        ${PROJECT_NAME}
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)