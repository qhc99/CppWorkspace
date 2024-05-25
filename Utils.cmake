cmake_minimum_required(VERSION 3.20)

set(CLANG_SANITIZERS_OPTIONS
    -fsanitize=leak
    -fsanitize=address
    -fsanitize=undefined
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls # perfect stack trace
)

set(CLANG_TEST_OPTIONS
    -Wall
    -fprofile-instr-generate
    -fcoverage-mapping
    -v
)

function(target_compile_link_options target visibility options)
    target_compile_options(${target} ${visibility} ${${options}})
    target_link_options(${target} ${visibility} ${${options}})
endfunction()

function(add_single_test single_file link_lib)
    set(name ${single_file})
    # Remove the .cpp suffix
    string(REGEX REPLACE "\\.cpp$" "" name ${name})
    # Transform CamelCase to lowercase with underscores
    string(REGEX REPLACE "([A-Z])" "_\\1" name ${name})
    string(TOLOWER ${name} name)
    # Remove leading underline
    string(REGEX REPLACE "^_" "" name ${name})
    
    message("---test name: " ${name} ", link_lib: " ${link_lib})
    message("---include path: " ${DOCTEST_INCLUDE_DIR})

    add_executable(${name} ${single_file})
    target_link_libraries(${name}
            PRIVATE
            ${link_lib}
            )
    target_include_directories(${name} PUBLIC ${DOCTEST_INCLUDE_DIR})
    
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_link_options(${name} PRIVATE CLANG_SANITIZERS_OPTIONS)
    endif()
    target_compile_link_options(${name} PRIVATE CLANG_TEST_OPTIONS)
    
    add_test(NAME ${name} COMMAND ${name})
endfunction()