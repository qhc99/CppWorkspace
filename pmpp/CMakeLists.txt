cmake_minimum_required(VERSION 3.28)

project(pmpp LANGUAGES CUDA)
find_package(doctest 2.4 CONFIG REQUIRED)

if(NOT SKIP_CUDA_ON_WINDOWS)
    add_library(vec_add_cu STATIC vec_add.cu vec_add.h utils.h)
    target_compile_options(vec_add_cu PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
    target_link_options(vec_add_cu PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
    set_target_properties(vec_add_cu PROPERTIES FOLDER "pmpp")
    add_unit_doctest(vec_add_test.cpp vec_add_cu "pmpp" "none" TRUE TRUE)

    add_executable(device_prop device_prop.cu vec_add.h utils.h)
    target_compile_options(device_prop PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
    target_link_options(device_prop PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
    set_target_properties(device_prop PROPERTIES FOLDER "pmpp")

    add_library(mat_mul_tiling STATIC mat_mul_tiling.cu mat_mul_tiling.h utils.h)
    target_compile_options(mat_mul_tiling PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
    target_link_options(mat_mul_tiling PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
    set_target_properties(mat_mul_tiling PROPERTIES FOLDER "pmpp")
    add_unit_doctest(mat_mul_tiling_test.cpp mat_mul_tiling "pmpp" "none" TRUE TRUE)
endif()