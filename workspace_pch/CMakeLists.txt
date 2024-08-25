cmake_minimum_required(VERSION 3.28)

project(workspace_pch LANGUAGES CXX)
add_library(workspace_pch INTERFACE)
target_precompile_headers(workspace_pch INTERFACE headers/workspace_pch.h)
target_include_directories(workspace_pch INTERFACE headers)

if(NVCC_COMPILER_PROGRAM AND NOT SKIP_CUDA_ON_WINDOWS)
    project(cuda_pch LANGUAGES CUDA)
    add_library(cuda_pch INTERFACE)
    target_precompile_headers(cuda_pch INTERFACE headers/cuda_pch.h)
    target_include_directories(cuda_pch INTERFACE headers)
endif  ()