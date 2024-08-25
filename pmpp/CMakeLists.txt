cmake_minimum_required(VERSION 3.28)

project(pmpp LANGUAGES CUDA)

if(NVCC_COMPILER_PROGRAM AND NOT SKIP_CUDA_ON_WINDOWS)
    find_package(doctest 2.4 CONFIG REQUIRED)

    add_library(vec_add_cu STATIC vec_add.cu vec_add.h utils.h)
    target_compile_options(vec_add_cu PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
    target_link_options(vec_add_cu PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
    set_target_properties(vec_add_cu PROPERTIES FOLDER "pmpp")
    add_unit_doctest(vec_add_test.cpp vec_add_cu "pmpp" "none" TRUE TRUE)

    add_executable(device_prop device_prop.cu vec_add.h utils.h)
    target_compile_options(device_prop PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
    target_link_options(device_prop PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
    set_target_properties(device_prop PROPERTIES FOLDER "pmpp")

    add_library(mat_mul_cu STATIC mat_mul.cu mat_mul.h utils.h)
    target_compile_options(mat_mul_cu PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
    target_link_options(mat_mul_cu PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
    set_target_properties(mat_mul_cu PROPERTIES FOLDER "pmpp")
    add_unit_doctest(mat_mul_test.cpp mat_mul_cu "pmpp" "none" TRUE TRUE)

    add_library(conv_cu STATIC conv.cu conv.h utils.h)
    target_compile_options(conv_cu PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
    target_link_options(conv_cu PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
    set_target_properties(conv_cu PROPERTIES FOLDER "pmpp")
    add_unit_doctest(conv_test.cpp conv_cu "pmpp" "none" TRUE TRUE)
endif()