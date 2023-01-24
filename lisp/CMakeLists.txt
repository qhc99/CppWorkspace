cmake_minimum_required(VERSION 3.20)

add_library(simple_lisp STATIC values.cpp values.h)
target_include_directories(simple_lisp PUBLIC pub)