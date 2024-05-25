cmake_minimum_required(VERSION 3.20)

include(Constant.cmake)

function(target_compile_link_options target visibility options)
    target_compile_options(${target} ${visibility} ${${options}})
    target_link_options(${target} ${visibility} ${${options}})
endfunction()

function(remove_dot_suffix str)
    # Remove everything after the first dot
    string(REGEX REPLACE "\\..*$" "" modified_string ${str})
    set(LATEST_RETURN ${modified_string} PARENT_SCOPE)
endfunction()

function(to_lowercase_underline str)
    # Transform CamelCase to lowercase with underscores
    string(REGEX REPLACE "([A-Z])" "_\\1" str ${str})
    string(TOLOWER ${str} str)
    # Remove leading underline
    string(REGEX REPLACE "^_" "" str ${str})
    set(LATEST_RETURN ${str} PARENT_SCOPE)
endfunction()

function(add_single_test single_file link_lib)
    remove_dot_suffix(${single_file})
    to_lowercase_underline(${LATEST_RETURN})
    set(name ${LATEST_RETURN})

    message("--- Add test name: " ${name} ", link_lib: " ${link_lib})
    message("--- Link include path: " ${DOCTEST_INCLUDE_DIR})

    add_executable(${name} ${single_file})
    target_link_libraries(${name}
        PRIVATE
        ${link_lib}
    )
    target_include_directories(${name} PUBLIC ${DOCTEST_INCLUDE_DIR})

    # Lib coverage and sanitizer options
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_link_options(${name} PRIVATE CLANG_SANITIZERS_OPTIONS)
        target_compile_link_options(${link_lib} PRIVATE CLANG_SANITIZERS_OPTIONS)
    endif()

    target_compile_link_options(${name} PRIVATE CLANG_TEST_OPTIONS)
    target_compile_link_options(${link_lib} PRIVATE CLANG_TEST_OPTIONS)

    # CTest intergration
    add_test(NAME ${name} COMMAND ${name})

    # Test coverage
    add_custom_target(run_${name}_coverage
        COMMAND ${CMAKE_COMMAND} -E echo "---Executable path: $<TARGET_FILE:${name}>"
        COMMAND $<TARGET_FILE:${name}>
        COMMAND llvm-profdata merge -sparse default.profraw -o temp.profdata
        COMMAND llvm-cov show -format=html -o ${CMAKE_SOURCE_DIR}/html_cov_report $<TARGET_FILE:${name}> -instr-profile="temp.profdata"
        COMMENT "---Test coverage output: ${CMAKE_SOURCE_DIR}/html_cov_report"
    )
endfunction()