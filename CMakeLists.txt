cmake_minimum_required(VERSION 3.28)



project(cpp_projects)

include(_cmake_libs/Utils.cmake)

enable_testing() # Enable at top level to use test UI

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_subdirectory(central_lib)
    message(STATUS ">>> Added lib central")
    
    add_subdirectory(misc)
    message(STATUS ">>> Added misc")
        
    add_subdirectory(chip8)
    message(STATUS ">>> Added chip8")
endif()

add_subdirectory(ray_trace)
