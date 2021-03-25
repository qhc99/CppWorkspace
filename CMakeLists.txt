cmake_minimum_required(VERSION 3.16)
project(learncpp)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g -O1")

add_executable(learncpp.exe src/main.cpp src/utils/utils.cpp src/utils/utils.h src/Euler/tools.h src/Euler/tools.cpp)
