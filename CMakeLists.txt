cmake_minimum_required(VERSION 3.16)

project(cpp_all_in_one)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    message(WARNING "NOT using Clang")
endif()

if(WIN32 AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(UNIX AND ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
        message(STATUS "Setting clang sanitizer flags")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")  # not well supported
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")  # not compatible to other flags
    endif()
endif()

add_subdirectory(src/learncpp)
add_subdirectory(src/numerics)
add_subdirectory(src/utils)
add_subdirectory(src/demos)
add_subdirectory(src/algo_cpp)
add_subdirectory(src/lisp)