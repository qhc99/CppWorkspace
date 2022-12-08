cmake_minimum_required(VERSION 3.20)

project(cpp_projects)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (UNIX)
        message(STATUS "Using GCC Sanitizers")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fsanitize=leak")
    endif ()
endif ()

add_subdirectory(src/main)
add_subdirectory(src/lib_central)
add_subdirectory(src/ray_trace)