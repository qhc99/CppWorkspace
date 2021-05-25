cmake_minimum_required(VERSION 3.16)

add_subdirectory(src)
if(NOT ${CROSS})
    add_subdirectory(tests)
endif()
