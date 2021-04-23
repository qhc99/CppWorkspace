cmake_minimum_required(VERSION 3.16)

project(cpp_all_in_one)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g -O1")

add_subdirectory(src/learncpp)
add_subdirectory(src/numerics)
add_subdirectory(src/utils)
add_subdirectory(src/demos)
add_subdirectory(src/algo_cpp)
add_subdirectory(src/lisp)