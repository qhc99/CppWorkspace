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
set(TARGET_LIB_DATA_STRUC_TEST data_struct_test)
add_executable(${TARGET_LIB_DATA_STRUC_TEST} DisjointSetTest.cpp RankSearchTest.cpp TernaryTriesTest.cpp)
target_link_libraries(${TARGET_LIB_DATA_STRUC_TEST}
        central_lib
        GTest::gtest_main
        )
# ASAN_OPTIONS=check_initialization_order=1
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(${TARGET_LIB_DATA_STRUC_TEST} PRIVATE ${COMMON_CLANG_SANITIZERS_COMPILE_LINK_OPTIONS_IN_TEST})
    target_link_options(${TARGET_LIB_DATA_STRUC_TEST} PRIVATE ${COMMON_CLANG_SANITIZERS_COMPILE_LINK_OPTIONS_IN_TEST})
endif()

target_compile_options(${TARGET_LIB_DATA_STRUC_TEST} PRIVATE ${COMMON_CLANG_COMPILE_LINK_OPTIONS_IN_TEST})
    target_link_options(${TARGET_LIB_DATA_STRUC_TEST} PRIVATE ${COMMON_CLANG_COMPILE_LINK_OPTIONS_IN_TEST})


include(GoogleTest)
gtest_discover_tests(${TARGET_LIB_DATA_STRUC_TEST})
