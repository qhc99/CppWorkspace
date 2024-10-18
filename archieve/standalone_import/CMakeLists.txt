cmake_minimum_required(VERSION 3.28)

project(standalone_import LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/_installed;${CMAKE_PREFIX_PATH}")

find_package(doctest 2 CONFIG REQUIRED)
find_package(CentralLib REQUIRED)

enable_testing()

add_executable(test_import test_import.cpp)
target_link_libraries(test_import
    PRIVATE
    CentralLib::central_lib_shared
    doctest::doctest
)
target_include_directories(test_import PRIVATE "${CMAKE_SOURCE_DIR}/_installed/include")
target_compile_definitions(test_import PRIVATE USE_SHARED)

add_test(NAME test_import COMMAND test_import)
# DLL dir path
set_property(TEST test_import PROPERTY ENVIRONMENT "Path=${CMAKE_SOURCE_DIR}/_installed/bin;$ENV{Path}" )