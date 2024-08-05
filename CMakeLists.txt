cmake_minimum_required(VERSION 3.28)

if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    message(FATAL_ERROR "Only Clang compiler is supported")
endif()

project(cpp_projects)

include(_cmake_libs/Utils.cmake)

enable_testing() # Enable at top level to use test UI

add_subdirectory(central_lib)
message(STATUS ">>> Added lib central")

add_subdirectory(misc)
message(STATUS ">>> Added misc")

add_subdirectory(ray_trace)

add_subdirectory(chip8)
message(STATUS ">>> Added chip8")