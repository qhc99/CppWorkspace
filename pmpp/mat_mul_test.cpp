#include <cstddef>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mat_mul.h"

TEST_CASE("mat_mul_tiling_small_mat_test"){
    size_t i = 3;
    size_t j = 3;
    size_t k = 8;
    auto* A = new float[i*j];
    auto* B = new float[j*k];
    auto* C = new float[i*k];

    // Use permutation matrix

    matMul(A, B, C, i, j, k);
    

    delete[] A;
    delete[] B;
    delete[] C;
}

TEST_CASE("mat_mul_tiling_large_mat_test")
{
    size_t i = 4096;
    size_t j = 4096;
    size_t k = 4096;
    auto* A = new float[i * j];
    auto* B = new float[j * k];
    auto* C = new float[i * k];

    // Use permutation matrix

    matMul(A, B, C, i, j, k);


    delete[] A;
    delete[] B;
    delete[] C;
}