cmake_minimum_required(VERSION 3.28)

find_program(LLVM_PROFDATA_PROGRAM llvm-profdata)
find_program(LLVM_COV_PROGRAM llvm-cov)
find_program(NVCC_COMPILER_PROGRAM nvcc)

if(CMAKE_SYSTEM_NAME MATCHES Windows AND (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC"))
    set(SKIP_CUDA_ON_WINDOWS TRUE)
else()
    set(SKIP_CUDA_ON_WINDOWS FALSE)
endif()