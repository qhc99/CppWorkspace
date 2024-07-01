cmake_minimum_required(VERSION 3.20)

set(link_lib central_lib)
add_single_test(TernaryTriesTest.cpp "${link_lib}")
add_single_test(RankSearchTest.cpp "${link_lib}")
add_single_test(DisjointSetTest.cpp "${link_lib}")
add_single_test(numeric_utils_test.cpp "${link_lib}")