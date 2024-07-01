cmake_minimum_required(VERSION 3.20)

project(cpp_projects)


include(Doctest.cmake)
include(Utils.cmake)

enable_testing() # Enable at top level to use test UI

add_subdirectory(central_lib)
print_info("Added lib central")

add_subdirectory(misc)
print_info("Added misc")

add_subdirectory(ray_trace)