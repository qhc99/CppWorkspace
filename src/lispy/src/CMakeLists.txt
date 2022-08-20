cmake_minimum_required(VERSION 3.16)
project(lisp_cpp)

add_library(${PROJECT_NAME}
        STATIC
        Lispy.cpp
        )
add_library(org::qhc::lisp ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC public PUBLIC private)