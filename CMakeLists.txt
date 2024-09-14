cmake_minimum_required(VERSION 3.28)

project(cpp_projects)

find_package(fmt 10 CONFIG REQUIRED)
find_package(doctest 2.4 CONFIG REQUIRED)

include(_cmake_libs/UtilsInclude.cmake)

enable_testing() # Enable at top level to use test UI

add_subdirectory(workspace_pch)
message(STATUS ">>> Added workspace pch")

add_subdirectory(central_lib)
message(STATUS ">>> Added lib central")

add_subdirectory(misc)
message(STATUS ">>> Added misc")
    
add_subdirectory(chip8)

add_subdirectory(ray_trace)

add_subdirectory(pmpp)
message(STATUS ">>> Added pmpp")

add_subdirectory(opengl_learn)
message(STATUS ">>> Added opengl projects")