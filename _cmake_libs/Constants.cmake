cmake_minimum_required(VERSION 3.20)

set(ASAN_OPTIONS
    -fsanitize=leak
    -fsanitize=address
    -fsanitize=undefined
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls # perfect stack trace
)

# Used in multithread programming
set(TSAN_OPTIONS
    -fsanitize=undefined
    -fsanitize=thread # not compatible with leak, address and memory
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
)

# Rarely used
set(MSAN_OPTIONS
    -fsanitize=undefined
    -fsanitize=memory # not compatible with leak, address and thread
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls 
)

set(WARN_ALL_OPTIONS 
    -Wall -Wextra -Wpedantic
)

set(TEST_COVERAGE_OPTIONS
    -fprofile-instr-generate
    -fcoverage-mapping
    -v
)

set(COMMON_OPTIONS
    ${WARN_ALL_OPTIONS}
)

set(COMMON_LINK_OPTIONS
    -detect-odr-violations
)