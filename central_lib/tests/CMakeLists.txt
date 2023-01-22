cmake_minimum_required(VERSION 3.20)
if (NOT DEFINED ENV{CUDACXX})
    #    find_package(Boost 1.74 REQUIRED COMPONENTS unit_test_framework)
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
    target_compile_options(data_struct_test PRIVATE
            -fsanitize=leak
            -fsanitize=address
            -fsanitize=undefined
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls
            -fsanitize-address-use-after-scope)
    target_link_options(data_struct_test PRIVATE
            -fsanitize=leak
            -fsanitize=address
            -fsanitize=undefined
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls
            -fsanitize-address-use-after-scope)

    include(GoogleTest)
    gtest_discover_tests(data_struct_test )
endif ()