cmake_minimum_required(VERSION 3.16)

project(lib_central_test)
enable_testing()

find_package(Boost 1.74.0 REQUIRED COMPONENTS unit_test_framework)

add_executable(DisjointSetTest DisjointSetTest.cpp)
target_link_libraries(DisjointSetTest
        dev::qhc::lib_central
        Boost::unit_test_framework
        )
target_include_directories(DisjointSetTest PRIVATE ${BOOST_INCLUDEDIR})

add_executable(RankSearchTest RankSearchTest.cpp)
target_link_libraries(RankSearchTest
        dev::qhc::lib_central
        Boost::unit_test_framework
        )
target_include_directories(RankSearchTest PRIVATE ${BOOST_INCLUDEDIR})
