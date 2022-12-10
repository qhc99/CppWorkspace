cmake_minimum_required(VERSION 3.20)

project(lib_central_test)
find_package(Boost 1.76)

if (Boost_FOUND)
    set(BOOST_INCLUDEDIR ${Boost_INCLUDE_DIRS})
    set(BOOST_LIBRARYDIR ${Boost_LIBRARY_DIRS})
else ()
    message(WARNING "cannot find boost, please specify BOOST_INCLUDEDIR and BOOST_LIBRARYDIR")
    set(BOOST_INCLUDEDIR $ENV{BOOST_INCLUDEDIR})
    set(BOOST_LIBRARYDIR $ENV{BOOST_LIBRARYDIR})
endif ()


message(STATUS "Find Boost include dir: ${BOOST_INCLUDEDIR}")
message(STATUS "Find Boost lib dir: ${BOOST_LIBRARYDIR}")

add_executable(DisjointSetTest DisjointSetTest.cpp)
target_link_libraries(DisjointSetTest
        dev::qhc::lib_central
        )
target_include_directories(DisjointSetTest PRIVATE ${BOOST_INCLUDEDIR})
target_link_directories(DisjointSetTest PRIVATE ${BOOST_LIBRARYDIR})

add_executable(RankSearchTest RankSearchTest.cpp)
target_link_libraries(RankSearchTest
        dev::qhc::lib_central
        )
target_include_directories(RankSearchTest PRIVATE ${BOOST_INCLUDEDIR})
target_link_directories(RankSearchTest PRIVATE ${BOOST_LIBRARYDIR})