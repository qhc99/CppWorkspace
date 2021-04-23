cmake_minimum_required(VERSION 3.16)

project(utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g -O1")

add_library(${PROJECT_NAME}
        STATIC
        utils.cpp
        miscellaneous.cpp
        )
add_library(org::nathan::utils ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC include)