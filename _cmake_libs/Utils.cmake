cmake_minimum_required(VERSION 3.28)

include(_cmake_libs/Constants.cmake)

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
# Arg: single_file, link_lib, folder_name, san_option, disable_coverage, no_libcmt
# Return generated test name $LATEST_RETURN
function(add_unit_doctest single_file link_lib folder_name)
    if(NOT DEFINED ARGV3)
        set(SAN_COMPILE_OPTIONS ASAN_COMPILE_OPTIONS)
        set(SAN_LINK_OPTIONS ASAN_LINK_OPTIONS)
    elseif(ARGV3 STREQUAL "asan")
        set(SAN_COMPILE_OPTIONS ASAN_COMPILE_OPTIONS)
        set(SAN_LINK_OPTIONS ASAN_LINK_OPTIONS)
    elseif(ARGV3 STREQUAL "tsan")
        if(MSVC)
            return()
        endif()
        set(SAN_COMPILE_OPTIONS TSAN_COMPILE_LINK_OPTIONS)
        set(SAN_LINK_OPTIONS TSAN_COMPILE_LINK_OPTIONS)
    elseif(ARGV3 STREQUAL "msan")
        if(MSVC)
            return()
        endif()
        set(SAN_COMPILE_OPTIONS MSAN_COMPILE_LINK_OPTIONS)
        set(SAN_LINK_OPTIONS MSAN_COMPILE_LINK_OPTIONS)
    elseif(ARGV3 STREQUAL "none")
        set(SAN_COMPILE_OPTIONS "")
        set(SAN_LINK_OPTIONS "")
    else()
        message(FATAL_ERROR "Arg 4 argument error: ${ARGV3}")
    endif()

    if(NOT DEFINED ARGV4)
        set(disable_test_coverage FALSE)        
    else()
        set(disable_test_coverage ${ARGV4})
    endif()

    if(NOT DEFINED ARGV5)
        set(no_libcmt FALSE)        
    else()
        set(no_libcmt ${ARGV5})
    endif()

    remove_dot_suffix(${single_file})
    to_lowercase_underline(${LATEST_RETURN})
    set(name ${LATEST_RETURN})
    set(LATEST_RETURN ${LATEST_RETURN} PARENT_SCOPE)

    message(STATUS ">>> Add test name: ${name}, link_lib: ${link_lib}")

    add_executable(${name} ${single_file})
    set_target_properties(${name} PROPERTIES FOLDER ${folder_name})
    target_link_libraries(${name}
        PRIVATE
        ${link_lib}
        doctest::doctest
    )

    # Lib coverage and sanitizer options
    target_compile_options(${name} PRIVATE $<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:${${SAN_COMPILE_OPTIONS}}> ${COMMON_COMPILE_OPTIONS} ${TEST_COVERAGE_OPTIONS})
    target_link_options(${name} PRIVATE $<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:${${SAN_LINK_OPTIONS}}> ${TEST_COVERAGE_OPTIONS} ${COMMON_LINK_OPTIONS} $<$<BOOL:${no_libcmt}>:$<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT>>)

    # CTest intergration
    add_test(NAME ${name} COMMAND ${name})

    if(NOT ${disable_test_coverage} AND LLVM_PROFDATA_EXIST AND LLVM_COV_EXIST AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Test coverage
        add_custom_target(run_${name}_coverage
            COMMAND ${CMAKE_COMMAND} -E echo "--- Executable path: $<TARGET_FILE:${name}>"
            COMMAND $<TARGET_FILE:${name}>
            COMMAND llvm-profdata merge -sparse default.profraw -o temp.profdata
            COMMAND llvm-cov show -format=html -o ${CMAKE_SOURCE_DIR}/_html_cov_report $<TARGET_FILE:${name}> -instr-profile="temp.profdata"
            COMMENT ">>> Test coverage output: ${CMAKE_SOURCE_DIR}/_html_cov_report"
        )
        set_target_properties(run_${name}_coverage PROPERTIES FOLDER ${folder_name}/test_coverage)
    endif()
endfunction()