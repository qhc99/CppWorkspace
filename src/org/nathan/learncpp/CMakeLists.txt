cmake_minimum_required(VERSION 3.16)

project(learncpp)
set(CMAKE_CXX_STANDARD 17)


add_executable(learncpp.exe src/main.cpp)

target_link_libraries(
        learncpp.exe
        org::nathan::demos
        org::nathan::Euler
        org::nathan::utils
        org::nathan::algo_cpp::structures
)
