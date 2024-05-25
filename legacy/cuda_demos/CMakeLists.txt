cmake_minimum_required(VERSION 3.22)
project(cuda_demos CUDA)

set(TARGET_HELLO_CUDA hello_CUDA)
add_executable(${TARGET_HELLO_CUDA} main.cu)
set_target_properties(${TARGET_HELLO_CUDA} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${TARGET_HELLO_CUDA} PRIVATE $<$<CONFIG:Debug>:-G>)
set_property(TARGET ${TARGET_HELLO_CUDA} PROPERTY CUDA_ARCHITECTURES OFF)

set(TARGET_ZERO_COPY zero_copy)
add_executable(${TARGET_ZERO_COPY} zero_copy.cu)
set_target_properties(${TARGET_ZERO_COPY} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${TARGET_ZERO_COPY} PRIVATE $<$<CONFIG:Debug>:-G>)
set_property(TARGET ${TARGET_ZERO_COPY} PROPERTY CUDA_ARCHITECTURES OFF)

