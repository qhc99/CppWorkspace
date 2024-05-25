cmake_minimum_required(VERSION 3.20)

project(lispcpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(lisp_test InputPortTest.cpp)
target_link_libraries(lisp_test
        simple_lisp
        )