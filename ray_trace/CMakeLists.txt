cmake_minimum_required(VERSION 3.28)

project(ray_trace LANGUAGES CXX)
SET(TARGET_RT_OMP rt_omp)
set(folder_name ray_race)
add_executable(${TARGET_RT_OMP}
    openmp/rt_omp.cpp
    openmp/lib.cpp
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
target_compile_options(${TARGET_RT_OMP} PRIVATE -fopenmp)
target_link_options(${TARGET_RT_OMP} PRIVATE -fopenmp)
set_property(TARGET ${TARGET_RT_OMP} PROPERTY CMAKE_CXX_STANDARD 20)
set_property(TARGET ${TARGET_RT_OMP} PROPERTY CMAKE_CXX_STANDARD_REQUIRED True)
set_target_properties(${TARGET_RT_OMP} PROPERTIES FOLDER ${folder_name})
message(STATUS ">>> Added ray trace omp")

find_program(NVCC_COMPILER_EXIST nvcc)

if(NVCC_COMPILER_EXIST)
    if(NOT CMAKE_BUILD_TYPE MATCHES Debug OR NOT CMAKE_SYSTEM_NAME MATCHES Windows)
        project(ray_trace_cuda LANGUAGES CUDA CXX)
        set(TARGET_RT_CUDA rt_cuda)
        set(CMAKE_CUDA_STANDARD 20)
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
        target_compile_options(${TARGET_RT_CUDA} PRIVATE $<$<CONFIG:Debug>:-G> -v) # Enable device code debug
        set_target_properties(${TARGET_RT_CUDA} PROPERTIES FOLDER ${folder_name})
        message(STATUS ">>> Added ray trace cuda")
    else()
        message(WARNING "Cuda debugging not supported in vscode on windows")
    endif()
endif()