cmake_minimum_required(VERSION 3.20)

project(cpp_projects)

set(CLANG_SANITIZERS_OPTIONS
    -fsanitize=leak
    -fsanitize=address
    -fsanitize=undefined
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls # perfect stack trace
)

set(CLANG_TEST_OPTIONS
    -Wall
    -fprofile-instr-generate
    -fcoverage-mapping
    -v
)

function(target_compile_link_options target visibility options)
    target_compile_options(${target} ${visibility} ${${options}})
    target_link_options(${target} ${visibility} ${${options}})
endfunction()

if(NOT DEFINED ENV{CUDACXX})
    add_subdirectory(central_lib)
    message(STATUS "Added lib central")

    # add_subdirectory(lisp)
    # message(STATUS "Added lisp interpreter")
    add_subdirectory(misc)
    message(STATUS "Added misc")
endif()

add_subdirectory(ray_trace)
