cmake_minimum_required(VERSION 3.20)

project(lib_central_test)
find_package(Boost 1.74 REQUIRED COMPONENTS unit_test_framework)


message(STATUS "Find Boost include dir: ${BOOST_INCLUDEDIR}")
message(STATUS "Find Boost lib dir: ${BOOST_LIBRARYDIR}")

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