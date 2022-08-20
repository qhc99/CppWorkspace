cmake_minimum_required(VERSION 3.16)

project(main_cpp)
add_executable(main main.cpp)

target_link_libraries(main
        org::qhc::lisp
        )


