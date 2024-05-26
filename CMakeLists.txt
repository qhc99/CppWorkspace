cmake_minimum_required(VERSION 3.20)

project(cpp_projects)


include(Doctest.cmake)
include(Utils.cmake)

if(NOT DEFINED ENV{CUDACXX})
    enable_testing() # Enable at top level to use test UI
    add_subdirectory(central_lib)
    message(STATUS "--- Added lib central")

    # add_subdirectory(lisp)
    # message(STATUS "--- Added lisp interpreter")
    add_subdirectory(misc)
    message(STATUS "--- Added misc")
endif()

add_subdirectory(ray_trace)