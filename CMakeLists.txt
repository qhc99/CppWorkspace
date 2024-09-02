cmake_minimum_required(VERSION 3.28)

project(cpp_projects)

find_package(fmt 10 CONFIG REQUIRED)
find_package(doctest 2.4 CONFIG REQUIRED)
find_package(SDL2 2.30 CONFIG REQUIRED)

include(_cmake_libs/Utils.cmake)

enable_testing() # Enable at top level to use test UI

add_subdirectory(workspace_pch)
message(STATUS ">>> Added workspace pch")

add_subdirectory(central_lib)
message(STATUS ">>> Added lib central")

add_subdirectory(misc)
message(STATUS ">>> Added misc")
    
add_subdirectory(chip8)
message(STATUS ">>> Added chip8")

add_subdirectory(ray_trace)

add_subdirectory(pmpp)
message(STATUS ">>> Added pmpp")

if(CMAKE_SYSTEM_NAME MATCHES Linux)
    add_subdirectory(x64_asm_linux)
    message(STATUS ">>> Added x64_asm_linux")
endif()