cmake_minimum_required(VERSION 3.16)

add_subdirectory(src)
if(UNIX)
    add_subdirectory(tests)
endif()
