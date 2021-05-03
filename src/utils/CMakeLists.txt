cmake_minimum_required(VERSION 3.16)

project(utils)

add_library(${PROJECT_NAME}
        STATIC
        utils.cpp
        miscellaneous.cpp
        )
add_library(org::nathan::utils ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC include)