cmake_minimum_required(VERSION 3.16)

project(miscellaneous_test)

add_executable(
        ${PROJECT_NAME}
        rank_search_test.cpp
)

target_link_libraries(
        ${PROJECT_NAME}
        org::nathan::algo_cpp::miscellaneous
        org::nathan::utils
        GTest::GTest
        GTest::Main
)