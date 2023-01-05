cmake_minimum_required(VERSION 3.20)



project(ray_trace)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD C17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD_REQUIRED True)

add_executable(rt rt.cpp vec3.h color.h color.h ray.h hittable.h sphere.h hittable_list.h rtweekend.h Camera.h material.h)

target_link_libraries(rt  dev::qhc::lib_central)
target_compile_options(rt PUBLIC -fopenmp)
target_link_options(rt PUBLIC -fopenmp)