cmake_minimum_required(VERSION 3.20)

project(main_cpp)

add_executable(rt rt.cpp vec3.h color.h color.h ray.h hittable.h sphere.h)

target_link_libraries(rt  dev::qhc::lib_central)
