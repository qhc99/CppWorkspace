cmake_minimum_required(VERSION 3.16)

project(cpp_all_in_one)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(UNIX)
        message(STATUS "Setting UNIX Clang sanitizer")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fsanitize=leak")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1 -g")
    endif()
endif()

add_subdirectory(src/learncpp)
add_subdirectory(src/numerics)
add_subdirectory(src/utils)
add_subdirectory(src/demos)
add_subdirectory(src/algo_cpp)
add_subdirectory(src/lisp)