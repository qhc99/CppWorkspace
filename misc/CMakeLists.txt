cmake_minimum_required(VERSION 3.28)

project(misc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(folder_name misc)

add_executable(locality_demo locality_demo.cpp)
target_link_libraries(locality_demo central_lib)
target_link_options(locality_demo PRIVATE -fopenmp)
target_compile_options(locality_demo PRIVATE -fopenmp)
set_target_properties(locality_demo PROPERTIES FOLDER ${folder_name})

add_executable(param_inherit param_inherit.cpp)
set_target_properties(param_inherit PROPERTIES FOLDER ${folder_name})

add_executable(co_generator co_generator.cpp)
set_target_properties(co_generator PROPERTIES FOLDER ${folder_name})

add_executable(interleve interleve.cpp)
set_target_properties(interleve PROPERTIES FOLDER ${folder_name})

add_executable(claire_voyant claire_voyant.cpp)
set_target_properties(claire_voyant PROPERTIES FOLDER ${folder_name})