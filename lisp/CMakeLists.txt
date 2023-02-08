cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)

add_library(simple_lisp
        STATIC
        values.cpp
        values.h 
        InputPort.cpp
        InputPort.h 
        Interpreter.cpp 
        Interpreter.h 
        exceptions.h)
target_include_directories(simple_lisp PUBLIC pub)

# add gtest
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/b796f7d44681514f58a683a3a71ff17c94edb0c1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(lisp_test InputPortTest.cpp)
target_link_libraries(lisp_test
        simple_lisp
        GTest::gtest_main
        )
target_compile_options(lisp_test PRIVATE
        -fsanitize=leak
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -fsanitize-address-use-after-scope

        --coverage
        )

target_link_options(lisp_test PRIVATE
        -fsanitize=leak
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -fsanitize-address-use-after-scope

        --coverage
        )

include(GoogleTest)

gtest_discover_tests(lisp_test)