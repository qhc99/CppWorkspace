cmake_minimum_required(VERSION 3.20)

project(lib_central)


#if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" AND UNIX AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
#        message(STATUS "Using GCC Sanitizers")
#        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
#        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fsanitize=leak")
#endif ()

add_library(${PROJECT_NAME}
        STATIC
        DisjointSet.cpp
        include/lib_central/numerics.h
        include/lib_central/rank_search.hpp
        include/lib_central/DisjointSet.h
        utils.cpp
        numerics.cpp)
add_library(dev::qhc::lib_central ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC include)


add_executable(locality_demo locality_demo.cpp)
target_link_libraries(locality_demo dev::qhc::lib_central)
