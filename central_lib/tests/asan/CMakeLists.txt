cmake_minimum_required(VERSION 3.28)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_unit_doctest(asan_leak_test.cpp ${link_lib} "central_libs_tests/asan" "asan" true)
    add_unit_doctest(asan_out_of_range_access_test.cpp ${link_lib}  "central_libs_tests/asan" "asan" true)
    add_unit_doctest(asan_use_after_free_test.cpp ${link_lib} "central_libs_tests/asan" "asan" true)
endif()

set(link_lib central_lib)
add_unit_doctest(asan_no_link_test.cpp ${link_lib} "central_libs_tests/asan" "none" true)