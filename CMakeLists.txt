cmake_minimum_required(VERSION 3.20)

project(cpp_projects)
if (NOT DEFINED ENV{CUDACXX})
    add_subdirectory(lib_central)
    message(STATUS "Added lib central")
    add_subdirectory(net_prog)
    message(STATUS "Added net prog")
    add_subdirectory(misc)
    message(STATUS "Added misc")
endif ()
add_subdirectory(ray_trace)
if (DEFINED ENV{CUDACXX})

    add_subdirectory(cuda_demos)
    message(STATUS "Added cuda demo")
endif ()

