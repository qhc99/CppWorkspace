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
    $<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>
)

set(ASAN_LINK_OPTIONS
    ${ASAN_CLANG_OPTIONS}
)

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
    $<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>
)

set(TEST_COVERAGE_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:-fprofile-instr-generate;-fcoverage-mapping>
)

set(COMMON_COMPILE_OPTIONS
    ${WARN_ALL_COMPILE_OPTIONS}
    $<$<CXX_COMPILER_ID:Clang>:$<$<BOOL:${COMPILE_LINK_VERBOSE}>:-v>;-std=c++20;$<$<BOOL:${CLANG_ARCH_NATIVE}>:-march=native>>
    $<$<CXX_COMPILER_ID:MSVC>:/std:c++20>
)

set(COMMON_LINK_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:$<$<BOOL:${COMPILE_LINK_VERBOSE}>:--verbose>>
    $<$<CXX_COMPILER_ID:MSVC>:$<$<BOOL:${COMPILE_LINK_VERBOSE}>:/VERBOSE>;/WX>
)

set(NVCC_COMMON_COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler "/W4 /WX /std:c++20" >
    $<$<CXX_COMPILER_ID:Clang>:-Xcompiler "${COMMON_COMPILE_OPTIONS}" >
    $<$<CXX_COMPILER_ID:Clang>:-Xlinker "${COMMON_LINK_OPTIONS}" >
    $<$<CONFIG:Debug>:-G> # Enable device code debug
    $<$<NOT:$<PLATFORM_ID:Windows>>:-ccbin=clang++;-Wno-gnu-line-marker>
    --std c++20
    $<$<BOOL:${COMPILE_LINK_VERBOSE}>:-v>
)

set(MSVC_NO_LIBCMT_LINK_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT> # Fix link warning
)

set(NVCC_COMMON_LINK_OPTIONS
    $<$<CXX_COMPILER_ID:MSVC>:${MSVC_NO_LIBCMT_LINK_OPTIONS};${COMMON_LINK_OPTIONS}>
)

set(COMPILE_OPENMP_OPTION $<$<CXX_COMPILER_ID:Clang>:-fopenmp> $<$<CXX_COMPILER_ID:MSVC>:/openmp>)

set(LINK_OPENMP_OPTION $<$<CXX_COMPILER_ID:Clang>:-fopenmp>)

set(SUPPRESS_COMPILE_WARNINGS_OPTION $<$<CXX_COMPILER_ID:Clang>:-w> $<$<CXX_COMPILER_ID:MSVC>:/w>)