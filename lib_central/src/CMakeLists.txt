cmake_minimum_required(VERSION 3.20)

add_library(${PROJECT_NAME}
        STATIC
        DisjointSet.cpp
        utils.cpp
        numerics.cpp)
add_library(dev::qhc::lib_central ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC public)


add_executable(locality_demo locality_demo.cpp)
target_link_libraries(locality_demo dev::qhc::lib_central)
