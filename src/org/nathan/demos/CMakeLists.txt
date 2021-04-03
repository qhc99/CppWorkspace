cmake_minimum_required(VERSION 3.16)

project(demos)
set(CMAKE_CXX_STANDARD 17)

add_library(
        ${PROJECT_NAME} STATIC
        src/randomDemo.cpp
        src/smartPtrAndIterDemo.cpp
        src/traitDemo.cpp
        src/tupleDestructionDemo.cpp
)

add_library(
        org::nathan::demos ALIAS ${PROJECT_NAME}
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)