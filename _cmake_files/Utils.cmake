cmake_minimum_required(VERSION 3.20)

include(_cmake_files/Constants.cmake)

function(target_compile_link_options target visibility options)
    target_compile_options(${target} ${visibility} ${${options}})
    target_link_options(${target} ${visibility} ${${options}})
endfunction()

function(print_info str)
    message(STATUS ">>> ${str}")
endfunction()

#
# Remove everything after the first dot
function(remove_dot_suffix str)
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

#
# Add single test file and library, generate test target and coverage target
#
# Return generated test name $LATEST_RETURN
function(add_single_test single_file link_lib)
    remove_dot_suffix(${single_file})
    to_lowercase_underline(${LATEST_RETURN})
    set(name ${LATEST_RETURN})
    set(LATEST_RETURN ${LATEST_RETURN} PARENT_SCOPE)

    print_info("Add test name: ${name}, link_lib: ${link_lib}")
    print_info("Link include path: ${DOCTEST_INCLUDE_DIR}")

    add_executable(${name} ${single_file})
    target_link_libraries(${name}
        PRIVATE
        ${link_lib}
    )
    target_include_directories(${name} PRIVATE ${DOCTEST_INCLUDE_DIR})

    # Lib coverage and sanitizer options
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_link_options(${name} PRIVATE CLANG_SANITIZERS_OPTIONS)
        # set options only for test
        target_compile_link_options(${link_lib} PRIVATE CLANG_SANITIZERS_OPTIONS)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_link_options(${name} PRIVATE CLANG_TEST_OPTIONS)
        target_compile_link_options(${link_lib} PRIVATE CLANG_TEST_OPTIONS)
    endif()

    # CTest intergration
    add_test(NAME ${name} COMMAND ${name})

    # Test coverage
    add_custom_target(run_${name}_coverage
        COMMAND ${CMAKE_COMMAND} -E echo "--- Executable path: $<TARGET_FILE:${name}>"
        COMMAND $<TARGET_FILE:${name}>
        COMMAND llvm-profdata merge -sparse default.profraw -o temp.profdata
        COMMAND llvm-cov show -format=html -o ${CMAKE_SOURCE_DIR}/_html_cov_report $<TARGET_FILE:${name}> -instr-profile="temp.profdata"
        COMMENT ">>> Test coverage output: ${CMAKE_SOURCE_DIR}/_html_cov_report"
    )
endfunction()