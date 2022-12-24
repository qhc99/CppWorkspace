cmake_minimum_required(VERSION 3.20)

find_package(Boost 1.74 REQUIRED COMPONENTS unit_test_framework)

add_executable(DisjointSetTest DisjointSetTest.cpp)
target_link_libraries(DisjointSetTest
        dev::qhc::lib_central
        Boost::unit_test_framework
        )

add_executable(RankSearchTest RankSearchTest.cpp)
target_link_libraries(RankSearchTest
        dev::qhc::lib_central
        Boost::unit_test_framework
        )