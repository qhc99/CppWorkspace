cmake_minimum_required(VERSION 3.28)

set(folder_name central_lib)
set(LIB_SOURCES
    DisjointSet.cpp
    utils.cpp
    numeric_utils.cpp
    public/lib_central/concepts_utils.h
    public/lib_central/DisjointSet.h
    public/lib_central/numeric_utils.h
    public/lib_central/rank_search.hpp
    public/lib_central/TernaryTries.hpp
    public/lib_central/utils.h
    )

add_library(central_lib STATIC ${LIB_SOURCES})
target_include_directories(central_lib PUBLIC public)
set_target_properties(central_lib PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(central_lib PRIVATE ${COMMON_LINK_OPTIONS})

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_library(central_lib_asan STATIC ${LIB_SOURCES})
    target_include_directories(central_lib_asan PUBLIC public)
    set_target_properties(central_lib_asan PROPERTIES FOLDER ${folder_name})
    target_compile_options(central_lib_asan PRIVATE ${ASAN_COMPILE_OPTIONS} ${COMMON_COMPILE_OPTIONS})
    target_link_options(central_lib_asan PRIVATE ${ASAN_LINK_OPTIONS} ${COMMON_LINK_OPTIONS})
endif()

find_package(fmt 10 CONFIG REQUIRED)
add_executable(
    lib_main
    main.cpp)
target_link_libraries(lib_main PRIVATE central_lib fmt::fmt-header-only)
set_target_properties(lib_main PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(central_lib PRIVATE ${COMMON_LINK_OPTIONS})