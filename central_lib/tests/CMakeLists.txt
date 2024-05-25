cmake_minimum_required(VERSION 3.20)

set(link_libs central_lib)
add_single_test(TernaryTriesTest.cpp "${link_libs}")
add_single_test(RankSearchTest.cpp "${link_libs}")
add_single_test(DisjointSetTest.cpp "${link_libs}")