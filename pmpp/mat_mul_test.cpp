#include <cstddef>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "mat_mul.h"
#include <doctest/doctest.h>

TEST_CASE("mat_mul_small_mat_test")
{
    constexpr size_t i = 3;
    constexpr size_t j = 3;
    constexpr size_t k = 8;
    auto* A = new float[i * j];
    auto* B = new float[j * k];
    auto* C = new float[i * k];
    std::fill(A, A + i * j, 0.f);
    std::iota(B, B + j * k, 0.f);
    // Use permutation matrix
    A[2] = 1;
    A[i + 1] = 1;
    A[2 * i] = 1;

    matMul(A, B, C, i, j, k);

    float counter {};
    for (int ti { i-1 }; ti >= 0; ti--) {
        for (size_t tk { 0 }; tk < k; tk++) {
            REQUIRE_EQ(counter++, C[ti * k + tk]);
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
}

TEST_CASE("mat_mul_large_mat_test")
{
    constexpr size_t i = 1024;
    constexpr size_t j = 1024;
    constexpr size_t k = 1024;
    auto* A = new float[i * j];
    auto* B = new float[j * k];
    auto* C = new float[i * k];
    std::fill(A, A + i * j, 0.f);
    std::iota(B, B + j * k, 0.f);
    // Use permutation matrix
    size_t tj_counter = j-1;
    for (size_t ti {}; ti < i; ti++) {
        A[ti*j+tj_counter--] = 1;
    }
    matMul(A, B, C, i, j, k);

    float counter {};
    for (int ti { i-1 }; ti >= 0; ti--) {
        for (size_t tk { 0 }; tk < k; tk++) {
            REQUIRE_EQ(counter++, C[ti * k + tk]);
        }
    }
    delete[] A;
    delete[] B;
    delete[] C;
}