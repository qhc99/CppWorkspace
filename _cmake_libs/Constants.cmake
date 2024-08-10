cmake_minimum_required(VERSION 3.28)

set(ASAN_CLANG_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
    $<$<NOT:$<PLATFORM_ID:Windows>>:-fsanitize=leak>
    $<$<PLATFORM_ID:Windows>:-shared-libsan>
    -fsanitize=address
    -fsanitize=undefined
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls # perfect stack trace
    >
)
set(ASAN_COMPILE_OPTIONS
    ${ASAN_CLANG_OPTIONS}
    $<$<CXX_COMPILER_ID:MSVC>:
    /fsanitize=address
    >
)

set(ASAN_LINK_OPTIONS ${ASAN_CLANG_OPTIONS})

# Used in multithread programming, currently not supported on windows
set(TSAN_COMPILE_LINK_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
    $<$<NOT:$<PLATFORM_ID:Windows>>:
    -fsanitize=undefined
    -fsanitize=thread # not compatible with leak, address and memory
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
    >
    >
)

# Detector of uninitialized memory use. Rarely used, currently not supported on windows
set(MSAN_COMPILE_LINK_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
    $<$<NOT:$<PLATFORM_ID:Windows>>:
    -fsanitize=undefined
    -fsanitize=memory # not compatible with leak, address and thread
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
    >
    >
)

# Enable all warnings and disable some warnings
set(WARN_ALL_COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -Wc++17-extensions>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX /analyze /std:c++17>
)

set(TEST_COVERAGE_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
    -fprofile-instr-generate
    -fcoverage-mapping
    >
)

set(COMMON_COMPILE_OPTIONS
    ${WARN_ALL_COMPILE_OPTIONS}
    $<$<CXX_COMPILER_ID:Clang>:-v>
)

set(COMMON_LINK_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:--verbose>
    $<$<CXX_COMPILER_ID:MSVC>:/VERBOSE>
)

find_program(LLVM_PROFDATA_EXIST llvm-profdata)
find_program(LLVM_COV_EXIST llvm-cov)