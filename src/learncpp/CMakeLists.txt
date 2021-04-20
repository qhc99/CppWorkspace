cmake_minimum_required(VERSION 3.16)

project(learncpp)
set(CMAKE_CXX_STANDARD 17)

add_executable(main.exe main.cpp)

target_link_libraries(main.exe
        org::nathan::utils
        org::nathan::demos
        )

