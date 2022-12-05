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
target_compile_definitions(DisjointSetTest PRIVATE "BOOST_TEST_DYN_LINK=1")

add_executable(RankSearchTest RankSearchTest.cpp)
target_link_libraries(RankSearchTest
        dev::qhc::lib_central
        Boost::unit_test_framework
        )
target_include_directories(RankSearchTest PRIVATE ${BOOST_INCLUDEDIR})
target_compile_definitions(RankSearchTest PRIVATE "BOOST_TEST_DYN_LINK=1")
