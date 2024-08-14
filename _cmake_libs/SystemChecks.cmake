cmake_minimum_required(VERSION 3.28)

find_program(LLVM_PROFDATA_EXIST llvm-profdata)
find_program(LLVM_COV_EXIST llvm-cov)
find_program(NVCC_COMPILER_EXIST nvcc)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    find_program(CL_COMPILER_EXIST cl)
    file(GLOB MSVC_DIR "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/*")
    # Get the first (or the most recent) version directory
    list(SORT MSVC_DIR ORDER DESCENDING)
    list(GET MSVC_DIR 0 MSVC_VERSION_DIR)
    set(MSVC_BIN_DIR "${MSVC_VERSION_DIR}/bin/Hostx64/x64")
    
    if((NOT EXISTS "${MSVC_BIN_DIR}") AND (NOT CL_COMPILER_EXIS))
        message(FATAL_ERROR "MSVC bin directory not found: ${MSVC_BIN_DIR}")
    endif()
endif()