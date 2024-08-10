cmake_minimum_required(VERSION 3.28)

add_unit_doctest(asan_leak_test.cpp "${link_lib}" "central_libs_tests/asan" "asan" true)
add_unit_doctest(asan_out_of_range_access_test.cpp "${link_lib}" "central_libs_tests/asan" "asan" true)
add_unit_doctest(asan_use_after_free_test.cpp "${link_lib}" "central_libs_tests/asan" "asan" true)

set(link_lib central_lib)
add_unit_doctest(no_asan_link_test.cpp "${link_lib}" "central_libs_tests/asan" "none" true)