cmake_minimum_required(VERSION 3.20)

project(cpp_projects)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "NOT using Clang")
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "NOT in linux env")
endif()

include(_cmake_libs/Doctest.cmake)
include(_cmake_libs/Utils.cmake)


enable_testing() # Enable at top level to use test UI

add_subdirectory(central_lib)
print_info("Added lib central")

add_subdirectory(misc)
print_info("Added misc")

add_subdirectory(ray_trace)