cmake_minimum_required(VERSION 3.20)

project(cpp_projects)

set(COMMON_CLANG_COMPILE_OPTIONS_IN_TEST 
    -Wall 
    -fsanitize=address 
    -fsanitize=undefined
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls # perfect stack trace
    --coverage
    )

if (NOT DEFINED ENV{CUDACXX})
    add_subdirectory(central_lib)
    message(STATUS "Added lib central")
    if (NOT WIN32)
        add_subdirectory(net_prog)
        message(STATUS "Added net prog")

    endif ()
    add_subdirectory(lisp)
    message(STATUS "Added lisp interpreter")
    add_subdirectory(misc)
    message(STATUS "Added misc")

else ()
    if (NOT WIN32)
        add_subdirectory(cuda_demos)
        message(STATUS "Added cuda demo")
    endif ()
endif ()
add_subdirectory(ray_trace)
message(STATUS "Added ray trace project")


