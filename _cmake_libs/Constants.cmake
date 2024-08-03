cmake_minimum_required(VERSION 3.20)

set(ASAN_OPTIONS
    $<$<PLATFORM_ID:Linux>:-fsanitize=leak>
    $<$<PLATFORM_ID:Windows>:-shared-libsan>
    -fsanitize=address
    -fsanitize=undefined
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls # perfect stack trace
)

# Used in multithread programming, not supported on windows
set(TSAN_OPTIONS
    -fsanitize=undefined
    -fsanitize=thread # not compatible with leak, address and memory
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
)

# Rarely used, not supported on windows
set(MSAN_OPTIONS
    -fsanitize=undefined
    -fsanitize=memory # not compatible with leak, address and thread
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
)

set(WARN_ALL_OPTIONS
    -Wall -Wextra -Wpedantic -Werror
)

set(TEST_COVERAGE_OPTIONS
    -fprofile-instr-generate
    -fcoverage-mapping
)

set(COMMON_OPTIONS
    ${WARN_ALL_OPTIONS}
    -v
)

set(COMMON_LINK_OPTIONS
    -detect-odr-violations
    -v
)