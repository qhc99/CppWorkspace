cmake_minimum_required(VERSION 3.28)

find_package(doctest 2.4 CONFIG REQUIRED)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(link_lib central_lib_asan)
else()
    set(link_lib central_lib)
endif()
add_unit_doctest(TernaryTriesTest.cpp ${link_lib} "central_libs_tests")
add_unit_doctest(rank_search_test.cpp ${link_lib} "central_libs_tests")
add_unit_doctest(DisjointSetTest.cpp ${link_lib} "central_libs_tests")
add_unit_doctest(numeric_utils_test.cpp ${link_lib} "central_libs_tests")

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/_installed" ${CMAKE_PREFIX_PATH})


add_subdirectory(asan)