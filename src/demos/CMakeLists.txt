cmake_minimum_required(VERSION 3.16)

project(demos)
set(CMAKE_CXX_STANDARD 17)

add_library(
        ${PROJECT_NAME} STATIC
        randomDemo.cpp
        smartPtrAndIterDemo.cpp
        traitDemo.cpp
        tupleDestructionDemo.cpp
        templateDemo.cpp
        importsTest.cpp
)

add_library(
        org::nathan::demos ALIAS ${PROJECT_NAME}
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        include
)

target_link_libraries(
        ${PROJECT_NAME}
        org::nathan::utils
        org::nathan::Euler
        org::nathan::algo_cpp::structures
        org::nathan::algo_cpp::miscellaneous
)