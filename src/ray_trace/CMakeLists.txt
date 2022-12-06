cmake_minimum_required(VERSION 3.16)

project(main_cpp)

add_executable(rt rt.cpp vec3.hpp)

target_link_libraries(rt  dev::qhc::lib_central)


