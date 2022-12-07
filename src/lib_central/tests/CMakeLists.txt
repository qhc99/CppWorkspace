cmake_minimum_required(VERSION 3.20)

project(lib_central_test)


message("find Boost include dir: $ENV{BOOST_INCLUDEDIR}")
message("find Boost lib dir: $ENV{BOOST_LIBRARYDIR}")

add_executable(DisjointSetTest DisjointSetTest.cpp)
target_link_libraries(DisjointSetTest
        dev::qhc::lib_central
        )
target_include_directories(DisjointSetTest PRIVATE $ENV{BOOST_INCLUDEDIR})
target_link_directories(DisjointSetTest PRIVATE $ENV{BOOST_LIBRARYDIR})

add_executable(RankSearchTest RankSearchTest.cpp)
target_link_libraries(RankSearchTest
        dev::qhc::lib_central
        )
target_include_directories(RankSearchTest PRIVATE $ENV{BOOST_INCLUDEDIR})
target_include_directories(RankSearchTest PRIVATE $ENV{BOOST_LIBRARYDIR})