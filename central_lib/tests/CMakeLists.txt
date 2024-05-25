cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD C20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD_REQUIRED True)


set(TARGET_LIB_DATA_STRUC_TEST tries_test)

add_executable(${TARGET_LIB_DATA_STRUC_TEST} TernaryTriesTest.cpp)
target_link_libraries(${TARGET_LIB_DATA_STRUC_TEST}
        central_lib
        )

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_link_options(${TARGET_LIB_DATA_STRUC_TEST} PRIVATE CLANG_SANITIZERS_OPTIONS)
endif()

target_compile_link_options(${TARGET_LIB_DATA_STRUC_TEST} PRIVATE CLANG_TEST_OPTIONS)
