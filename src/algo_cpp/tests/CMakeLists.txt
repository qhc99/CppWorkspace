cmake_minimum_required(VERSION 3.16)

project(algo_cpp_test)

get_filename_component(test2 DisjointSetTest.cpp NAME_WLE)
add_executable(
        ${test2}
        DisjointSetTest.cpp
)
target_link_libraries(
        ${test2}
        org::nathan::algo_cpp::structures
        GTest::GTest
        GTest::Main
)

get_filename_component(test1 rank_search_text.cpp NAME_WLE)
add_executable(
        ${test1}
        rank_search_test.cpp
)
target_link_libraries(
        ${test1}
        org::nathan::algo_cpp::miscellaneous
        org::nathan::utils
        GTest::GTest
        GTest::Main
)

