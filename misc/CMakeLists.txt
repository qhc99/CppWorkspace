cmake_minimum_required(VERSION 3.28)

project(misc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(folder_name misc)

add_executable(locality_demo locality_demo.cpp)
target_link_libraries(locality_demo central_lib workspace_pch)
target_compile_options(locality_demo PRIVATE $<$<CXX_COMPILER_ID:Clang>:-fopenmp;-Wno-pass-failed> $<$<CXX_COMPILER_ID:MSVC>:/openmp>)
target_link_options(locality_demo PRIVATE $<$<CXX_COMPILER_ID:Clang>:-fopenmp>)
set_target_properties(locality_demo PROPERTIES FOLDER ${folder_name})

add_executable(param_inherit param_inherit.cpp)
target_link_libraries(param_inherit PRIVATE workspace_pch)
set_target_properties(param_inherit PROPERTIES FOLDER ${folder_name})

add_executable(co_generator co_generator.cpp)
target_link_libraries(co_generator PRIVATE workspace_pch)
set_target_properties(co_generator PROPERTIES FOLDER ${folder_name})

add_executable(interleve interleve.cpp)
target_link_libraries(interleve PRIVATE workspace_pch)
set_target_properties(interleve PROPERTIES FOLDER ${folder_name})

add_executable(claire_voyant claire_voyant.cpp)
target_link_libraries(claire_voyant PRIVATE workspace_pch)
set_target_properties(claire_voyant PROPERTIES FOLDER ${folder_name})