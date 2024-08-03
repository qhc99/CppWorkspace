cmake_minimum_required(VERSION 3.20)

set(folder_name central_lib)
set(LIB_SOURCES
    DisjointSet.cpp
    utils.cpp
    numeric_utils.cpp)

add_library(central_lib STATIC ${LIB_SOURCES})
add_library(central_lib_asan STATIC ${LIB_SOURCES})

target_include_directories(central_lib PUBLIC public)
target_include_directories(central_lib_asan PUBLIC public)

set_target_properties(central_lib PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib PRIVATE ${COMMON_OPTIONS})
target_link_options(central_lib PRIVATE ${COMMON_OPTIONS} ${COMMON_LINK_OPTIONS})

set_target_properties(central_lib_asan PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib_asan PRIVATE ${ASAN_OPTIONS} ${COMMON_OPTIONS})
target_link_options(central_lib_asan PRIVATE ${ASAN_OPTIONS} ${COMMON_OPTIONS} ${COMMON_LINK_OPTIONS})

find_package(fmt CONFIG REQUIRED)
add_executable(
    lib_main
    main.cpp)
target_link_libraries(lib_main PRIVATE central_lib fmt::fmt)
set_target_properties(lib_main PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib PRIVATE ${COMMON_OPTIONS})
target_link_options(central_lib PRIVATE ${COMMON_OPTIONS} ${COMMON_LINK_OPTIONS})