cmake_minimum_required(VERSION 3.20)

add_library(simple_lisp
        STATIC
        values.cpp
        values.h lisp.cpp
        InputPort.cpp
        InputPort.h)
target_include_directories(simple_lisp PUBLIC pub)