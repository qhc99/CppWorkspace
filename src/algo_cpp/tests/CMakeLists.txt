cmake_minimum_required(VERSION 3.16)

project(algo_cpp_test)

enable_testing()

get_filename_component(test1 rank_search_text.cpp NAME_WLE)
add_executable(
        ${test1}
        RankSearchTest.cpp
)
target_link_libraries(
        ${test1}
        org::nathan::algo_cpp::miscellaneous
        org::nathan::utils
        GTest::GTest
        GTest::Main
)

get_filename_component(test2 DisjointSetTest.cpp NAME_WLE)
add_executable(
        ${test2}
        DisjointSetTest.cpp
        RankSearchTest.cpp
)
target_link_libraries(
        ${test2}
        org::nathan::algo_cpp::structures
        org::nathan::utils
        GTest::GTest
        GTest::Main
)



