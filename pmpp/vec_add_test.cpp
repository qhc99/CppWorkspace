#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "vec_add.h"

TEST_CASE("vec_add_test")
{
    int size = 3;
    float* A = new float[size];
    float* B = new float[size];
    float* C = new float[size];
    A[0] = 1;
    B[0] = 1;
    A[1] = 2;
    B[1] = 2;
    A[2] = 3;
    B[2] = 3;
    vecAdd(A, B, C, size);
    REQUIRE(C[0] == 2);
    REQUIRE(C[1] == 4);
    REQUIRE(C[2] == 6);
    delete[] A;
    delete[] B;
    delete[] C;
}