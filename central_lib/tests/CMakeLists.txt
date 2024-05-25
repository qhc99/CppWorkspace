cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD C20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD_REQUIRED True)

function(add_single_test name single_file)
    add_executable(${name} ${single_file})
    target_link_libraries(${name}
            central_lib
            )
    
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_link_options(${name} PRIVATE CLANG_SANITIZERS_OPTIONS)
    endif()
    target_compile_link_options(${name} PRIVATE CLANG_TEST_OPTIONS)
    
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_single_test(tries_test TernaryTriesTest.cpp)
add_single_test(rank_search_test RankSearchTest.cpp)
add_single_test(disjoint_set_test DisjointSetTest.cpp)