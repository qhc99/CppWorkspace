cmake_minimum_required(VERSION 3.16)

project(algo_cpp_miscellaneous)
set(CMAKE_CXX_STANDARD 17)


add_library(
        ${PROJECT_NAME}
        INTERFACE
)

add_library(
        org::nathan::algo_cpp::miscellaneous
        ALIAS
        ${PROJECT_NAME}
)

target_include_directories(
        ${PROJECT_NAME}
        INTERFACE
        include
)