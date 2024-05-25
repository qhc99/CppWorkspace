cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD C20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/b796f7d44681514f58a683a3a71ff17c94edb0c1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(data_struct_test DisjointSetTest.cpp RankSearchTest.cpp TernaryTriesTest.cpp)
target_link_libraries(data_struct_test
        central_lib
        GTest::gtest_main
        )
# ASAN_OPTIONS=check_initialization_order=1
target_compile_options(data_struct_test PRIVATE
        # -fsanitize=address
        # -fsanitize=undefined
        # -fno-omit-frame-pointer
        # -fno-optimize-sibling-calls # perfect stack trace
        --coverage
        )
target_link_options(data_struct_test PRIVATE
        # -fsanitize=address
        # -fsanitize=undefined
        # -fno-omit-frame-pointer
        # -fno-optimize-sibling-calls # perfect stack trace
        --coverage
        )

include(GoogleTest)
gtest_discover_tests(data_struct_test)
