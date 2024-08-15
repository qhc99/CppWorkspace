cmake_minimum_required(VERSION 3.28)

# TODO:
# nvcc: --compiler-options, --linker-options, --compiler-bindir
# test dylib
find_package(doctest 2.4 CONFIG REQUIRED)
add_library(vec_add_cu STATIC vec_add.cu vec_add.h utils.h)
target_compile_options(vec_add_cu  PRIVATE ${NVCC_COMMON_COMPILE_OPTIONS})
target_link_options(vec_add_cu PRIVATE ${NVCC_COMMON_LINK_OPTIONS})
set_target_properties(vec_add_cu PROPERTIES FOLDER "pmpp/example")

add_unit_doctest(vec_add_test.cpp vec_add_cu "pmpp/example" "none" TRUE TRUE)