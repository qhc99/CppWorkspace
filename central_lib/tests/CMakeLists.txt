cmake_minimum_required(VERSION 3.20)

set(link_lib central_lib)
add_single_lib_test(TernaryTriesTest.cpp "${link_lib}" "central_libs_tests")
add_single_lib_test(rank_search_test.cpp "${link_lib}" "central_libs_tests")
add_single_lib_test(DisjointSetTest.cpp "${link_lib}" "central_libs_tests")
add_single_lib_test(numeric_utils_test.cpp "${link_lib}" "central_libs_tests")
