cmake_minimum_required(VERSION 3.20)

set(CLANG_SANITIZERS_OPTIONS
    -fsanitize=leak
    -fsanitize=address
    -fsanitize=undefined
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls # perfect stack trace
)

# Use in multithread programming
set(CLANG_THREAD_SANITIZERS_OPTIONS
    -fsanitize=undefined
    -fsanitize=thread # not compatible with leak, address and memory
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls 
)

# Rarely use
set(CLANG_MEMORY_SANITIZERS_OPTIONS
    -fsanitize=undefined
    -fsanitize=memory # not compatible with leak, address and thread
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls 
)

set(CLANG_TEST_OPTIONS
    -Wall
    -fprofile-instr-generate
    -fcoverage-mapping
    -v
)