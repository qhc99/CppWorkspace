cmake_minimum_required(VERSION 3.20)

project(misc)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD C20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD_REQUIRED True)

add_executable(locality_demo locality_demo.cpp)
target_link_libraries(locality_demo central_lib)
target_link_options(locality_demo PUBLIC -fopenmp)
target_compile_options(locality_demo PUBLIC -fopenmp)
