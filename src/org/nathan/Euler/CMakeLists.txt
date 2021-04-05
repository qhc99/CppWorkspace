cmake_minimum_required(VERSION 3.16)

project(Euler)

add_library(
        ${PROJECT_NAME}
        STATIC
        src/numerics.cpp
)

add_library(
        org::nathan::Euler
        ALIAS
        ${PROJECT_NAME}
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)