cmake_minimum_required(VERSION 3.16)

project(algo_cpp_test)
find_package(Boost 1.71.0 REQUIRED COMPONENTS unit_test_framework)
enable_testing()

add_executable(DisjointSetTest DisjointSetTest.cpp)
target_link_libraries(DisjointSetTest
        org::nathan::algo_cpp
        Boost::unit_test_framework
        )
target_compile_definitions(DisjointSetTest PRIVATE BOOST_TEST_DYN_LINK)

add_executable(RankSearchTest RankSearchTest.cpp)
target_link_libraries(RankSearchTest
        org::nathan::algo_cpp
        org::nathan::utils
        Boost::unit_test_framework
        )
target_compile_definitions(RankSearchTest PRIVATE BOOST_TEST_DYN_LINK)

#add_test(test_all_algo_cpp DisjointSetTest)




