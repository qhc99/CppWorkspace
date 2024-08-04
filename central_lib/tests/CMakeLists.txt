cmake_minimum_required(VERSION 3.28)

set(link_lib central_lib_asan)
add_unit_test(TernaryTriesTest.cpp "${link_lib}" "central_libs_tests")
add_unit_test(rank_search_test.cpp "${link_lib}" "central_libs_tests")
add_unit_test(DisjointSetTest.cpp "${link_lib}" "central_libs_tests")
add_unit_test(numeric_utils_test.cpp "${link_lib}" "central_libs_tests")

add_subdirectory(asan)