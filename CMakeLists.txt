cmake_minimum_required(VERSION 3.20)

project(cpp_projects)



if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (UNIX)
        message(STATUS "Using GCC Sanitizers")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fsanitize=leak")
    endif ()
endif ()

add_subdirectory(lib_central)
add_subdirectory(ray_trace)
