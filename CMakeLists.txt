cmake_minimum_required(VERSION 3.16)
project(learncpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -g -O1")

add_executable(learncpp src/main.cpp src/randomDemo.cpp src/traitDemo.cpp src/smartPtrAndIterDemo.cpp src/smartPtrAndIterDemo.h src/traitDemo.h src/randomDemo.h src/utils.cpp src/utils.h)