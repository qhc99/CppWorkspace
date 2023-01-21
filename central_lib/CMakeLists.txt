cmake_minimum_required(VERSION 3.20)

project(central_lib)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD C20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD_REQUIRED True)

add_subdirectory(src)
add_subdirectory(tests)
