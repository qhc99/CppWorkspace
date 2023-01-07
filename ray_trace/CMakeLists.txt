cmake_minimum_required(VERSION 3.20)

if (NOT DEFINED ENV{CUDACXX})
    message(STATUS "Added ray trace")
    project(ray_trace CXX)
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

    target_link_libraries(${TARGET_RT_OMP} dev::qhc::lib_central)
    target_compile_options(${TARGET_RT_OMP} PUBLIC -fopenmp)
    target_link_options(${TARGET_RT_OMP} PUBLIC -fopenmp)
    set_property(TARGET ${TARGET_RT_OMP} PROPERTY CMAKE_CXX_STANDARD 20)
    set_property(TARGET ${TARGET_RT_OMP} PROPERTY CMAKE_CXX_STANDARD_REQUIRED True)
endif ()

if (DEFINED ENV{CUDACXX})
    message(STATUS "Added ray trace cuda")
    project(ray_trace_cuda CUDA)
    set(TARGET_RT_CUDA rt_cuda)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    add_executable(${TARGET_RT_CUDA}
            cuda/Vec3.h
            cuda/write_color.h
            cuda/write_color.h
            cuda/Ray.h
            cuda/Hittable.h
            cuda/Sphere.h
            cuda/HittableList.h
            cuda/rtweekend.h
            cuda/Camera.h
            cuda/Material.h
            cuda/rt_cuda.cu
            cuda/utils.h)
    set_target_properties(${TARGET_RT_CUDA} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${TARGET_RT_CUDA} PRIVATE $<$<CONFIG:Debug>:-G>)
    set_property(TARGET ${TARGET_RT_CUDA} PROPERTY CUDA_ARCHITECTURES OFF)
endif ()