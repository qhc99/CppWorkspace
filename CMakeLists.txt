cmake_minimum_required(VERSION 3.16)
project(learncpp)

find_package(GTest CONFIG REQUIRED)

file(GLOB SOURCES "src/*.cpp" "src/*/*.cpp")


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g -O1")

add_executable(learncpp ${SOURCES} src/demos/tupleDestructionDemo.cpp src/demos/tupleDestructiondemo.h)
