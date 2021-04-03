cmake_minimum_required(VERSION 3.16)

project(learncpp)

add_executable(learncpp.exe src/main.cpp)

target_link_libraries(
        learncpp.exe
        org::nathan::demos
        org::nathan::Euler
        org::nathan::utils
)
