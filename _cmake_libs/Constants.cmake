cmake_minimum_required(VERSION 3.20)

set(ASAN_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
        $<$<STREQUAL:$<PLATFORM_ID>,Linux>: -fsanitize=leak>
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls # perfect stack trace
        >
    $<$<AND:$<BOOL:${MSVC}>,$<STREQUAL:$<PLATFORM_ID>,Windows>>:/fsanitize=address>
)

# Used in multithread programming, currently not supported on windows
set(TSAN_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
        -fsanitize=undefined
        -fsanitize=thread # not compatible with leak, address and memory
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
    > 
)

# Rarely used, currently not supported on windows
set(MSAN_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
        -fsanitize=undefined
        -fsanitize=memory # not compatible with leak, address and thread
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls 
    >
)

set(WARN_ALL_OPTIONS 
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -Werror>
    $<$<AND:$<BOOL:${MSVC}>,$<STREQUAL:$<PLATFORM_ID>,Windows>>:/W4 /WX>
)

set(TEST_COVERAGE_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
        -fprofile-instr-generate
        -fcoverage-mapping
        -v
    >
)

set(COMMON_OPTIONS
    ${WARN_ALL_OPTIONS}
    $<$<CONFIG:Debug>:
        $<$<CXX_COMPILER_ID:Clang>:-G>
        $<$<AND:$<BOOL:${MSVC}>,$<STREQUAL:$<PLATFORM_ID>,Windows>>:/DEBUG>
    >
)

set(COMMON_LINK_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:-detect-odr-violations>
)