#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mat_mul_tiling.h"

TEST_CASE("mat_mul_tiling_test"){
    int i = 3, j = 3, k = 5;
    float* A = new float[i*j];
    float* B = new float[j*k];
    float* C = new float[i*k];

    // Use permutation matrix

    matMulTiling(A, B, C, i, j, k);
    
    REQUIRE(C[0] == 2);

    delete[] A;
    delete[] B;
    delete[] C;
}