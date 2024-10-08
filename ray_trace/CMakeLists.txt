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

target_link_libraries(${TARGET_RT_OMP} PRIVATE central_lib workspace_pch)
target_compile_options(${TARGET_RT_OMP} PRIVATE $<$<CXX_COMPILER_ID:Clang>:-std=c++20> $<$<CXX_COMPILER_ID:MSVC>:/std:c++20> ${COMPILE_OPENMP_OPTION})
target_link_options(${TARGET_RT_OMP} PRIVATE ${LINK_OPENMP_OPTION})
set_target_properties(${TARGET_RT_OMP} PROPERTIES FOLDER ${folder_name})
message(STATUS ">>> Added ray trace omp")

if(NVCC_COMPILER_PROGRAM AND NOT SKIP_CLANG_WINDOWS_CUDA)
    project(ray_trace_cuda LANGUAGES CUDA)
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
    target_compile_options(${TARGET_RT_CUDA} PRIVATE
        $<$<CONFIG:Debug>:-G> # Enable device code debug
        $<$<NOT:$<PLATFORM_ID:Windows>>:-ccbin=clang++>
        --std=c++20
        $<$<BOOL:${COMPILE_LINK_VERBOSE}>:-v>
    )
    target_link_libraries(${TARGET_RT_CUDA} PRIVATE workspace_pch)
    target_link_options(${TARGET_RT_CUDA} PRIVATE $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:MSVC>>:/NODEFAULTLIB:LIBCMT>) # Fix link warning
    set_target_properties(${TARGET_RT_CUDA} PROPERTIES FOLDER ${folder_name})
    message(STATUS ">>> Added ray trace cuda")
endif()