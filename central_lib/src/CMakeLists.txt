cmake_minimum_required(VERSION 3.28)

find_package(fmt 10 CONFIG REQUIRED)

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
target_link_libraries(central_lib PRIVATE workspace_pch)


add_library(central_lib_shared SHARED ${LIB_SOURCES})
target_include_directories(central_lib_shared PUBLIC public)
set_target_properties(central_lib_shared PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib_shared PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(central_lib_shared PRIVATE ${COMMON_LINK_OPTIONS})
target_link_libraries(central_lib_shared PRIVATE workspace_pch)
target_compile_definitions(central_lib_shared PRIVATE BUILDING_CENTRAL_LIB)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_library(central_lib_asan STATIC ${LIB_SOURCES})
    target_include_directories(central_lib_asan PUBLIC public)
    set_target_properties(central_lib_asan PROPERTIES FOLDER ${folder_name})
    target_compile_options(central_lib_asan PRIVATE ${ASAN_COMPILE_OPTIONS} ${COMMON_COMPILE_OPTIONS})
    target_link_options(central_lib_asan PRIVATE ${ASAN_LINK_OPTIONS} ${COMMON_LINK_OPTIONS})
    target_link_libraries(central_lib_asan PRIVATE workspace_pch)
endif()

add_executable(
    lib_main
    main.cpp)
target_link_libraries(lib_main PRIVATE central_lib fmt::fmt-header-only workspace_pch)
set_target_properties(lib_main PROPERTIES FOLDER ${folder_name})


add_executable(
    shared_link_demo
    shared_link.cpp)
target_link_libraries(shared_link_demo PRIVATE central_lib_shared fmt::fmt-header-only workspace_pch)
set_target_properties(shared_link_demo PROPERTIES FOLDER ${folder_name})