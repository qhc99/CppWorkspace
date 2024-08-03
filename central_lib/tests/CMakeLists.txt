cmake_minimum_required(VERSION 3.20)

set(link_lib central_lib_asan)
add_single_lib_test(TernaryTriesTest.cpp "${link_lib}" "central_libs_tests")
add_single_lib_test(rank_search_test.cpp "${link_lib}" "central_libs_tests")
add_single_lib_test(DisjointSetTest.cpp "${link_lib}" "central_libs_tests")
add_single_lib_test(numeric_utils_test.cpp "${link_lib}" "central_libs_tests")
add_single_lib_test(asan_leak_test.cpp "${link_lib}" "central_libs_tests" true)
add_single_lib_test(asan_out_of_range_access_test.cpp "${link_lib}" "central_libs_tests" true)
add_single_lib_test(asan_use_after_free_test.cpp "${link_lib}" "central_libs_tests" true)


set(link_lib central_lib)
add_single_lib_test(no_asan_link_test.cpp "${link_lib}" "central_libs_tests" true 0)
