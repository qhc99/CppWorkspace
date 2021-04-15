cmake_minimum_required(VERSION 3.16)

project(algo_cpp_structures_test)
set(CMAKE_CXX_STANDARD 17)


add_executable(
        ${PROJECT_NAME}
        DisjointSetTest.cpp
)

target_link_libraries(
        ${PROJECT_NAME}
        org::nathan::algo_cpp::structures
        GTest::GTest
        GTest::Main
)

