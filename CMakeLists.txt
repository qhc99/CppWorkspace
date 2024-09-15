cmake_minimum_required(VERSION 3.28)

project(cpp_projects)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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