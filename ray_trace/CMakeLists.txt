cmake_minimum_required(VERSION 3.20)

print_info("Added ray trace omp")
project(ray_trace LANGUAGES CXX)
SET(TARGET_RT_OMP rt_omp)
add_executable(${TARGET_RT_OMP}
        openmp/rt_omp.cpp
        openmp/vec3.h
        openmp/color.h
        openmp/color.h
        openmp/ray.h
        openmp/hittable.h
        openmp/sphere.h
        openmp/hittable_list.h
        openmp/rtweekend.h
        openmp/camera.h
        openmp/material.h)

target_link_libraries(${TARGET_RT_OMP} central_lib)
target_compile_options(${TARGET_RT_OMP} PUBLIC -fopenmp)
target_link_options(${TARGET_RT_OMP} PUBLIC -fopenmp)
set_property(TARGET ${TARGET_RT_OMP} PROPERTY CMAKE_CXX_STANDARD 17)
set_property(TARGET ${TARGET_RT_OMP} PROPERTY CMAKE_CXX_STANDARD_REQUIRED True)

if (DEFINED ENV{CUDACXX} AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    print_info("Added ray trace cuda")
    project(ray_trace_cuda LANGUAGES CUDA CXX)
    set(TARGET_RT_CUDA rt_cuda)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    add_executable(${TARGET_RT_CUDA}
            cuda/camera.h
            cuda/hitable.h
            cuda/hitable_list.h
            cuda/main.cu
            cuda/material.h
            cuda/main.cu
            cuda/ray.h
            cuda/sphere.h
            cuda/vec3.h)
    set_target_properties(${TARGET_RT_CUDA} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${TARGET_RT_CUDA} PRIVATE $<$<CONFIG:Debug>:-G>)
    set_property(TARGET ${TARGET_RT_CUDA} PROPERTY CUDA_ARCHITECTURES OFF)
endif ()