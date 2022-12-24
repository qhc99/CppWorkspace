cmake_minimum_required(VERSION 3.20)

project(lib_central)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD C23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(src)
add_subdirectory(tests)
