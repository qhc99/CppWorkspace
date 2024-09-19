cmake_minimum_required(VERSION 3.28)

project(standalone_import LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
find_package(doctest 2.4 CONFIG REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/_installed")
find_package(CentralLib)

add_executable(test_import test_import.cpp)
target_compile_options(test_import PRIVATE     
$<$<CXX_COMPILER_ID:Clang>:-std=c++20>
$<$<CXX_COMPILER_ID:MSVC>:/std:c++20>)
target_link_libraries(test_import
    PRIVATE
    CentralLib::central_lib_shared
    doctest::doctest
)
target_include_directories(test_import PRIVATE "${CMAKE_SOURCE_DIR}/_installed/include")
target_compile_definitions(test_import PRIVATE USE_SHARED)