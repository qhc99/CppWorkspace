#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "conv2d.h"
#include <doctest/doctest.h>

void conv2dOpenmp(float* N, float* F, float* P, int radius, int width, int height);

TEST_CASE("conv2d_small_test")
{

    float N[9];
    float F[9];
    float P[9];
    float P_cmp[9];

    std::fill(N, N + 9, 1);
    std::fill(F, F + 9, 1);

    conv2d(N, F, P, 1, 3, 3);
    conv2dOpenmp(N, F, P_cmp, 1, 3, 3);

    REQUIRE_EQ(4, P[0]);
    REQUIRE_EQ(6, P[1]);
    REQUIRE_EQ(4, P[2]);
    REQUIRE_EQ(6, P[3]);
    REQUIRE_EQ(9, P[4]);
    REQUIRE_EQ(6, P[5]);
    REQUIRE_EQ(4, P[6]);
    REQUIRE_EQ(6, P[7]);
    REQUIRE_EQ(4, P[8]);

    REQUIRE_EQ(P_cmp[0], P[0]);
    REQUIRE_EQ(P_cmp[1], P[1]);
    REQUIRE_EQ(P_cmp[2], P[2]);
    REQUIRE_EQ(P_cmp[3], P[3]);
    REQUIRE_EQ(P_cmp[4], P[4]);
    REQUIRE_EQ(P_cmp[5], P[5]);
    REQUIRE_EQ(P_cmp[6], P[6]);
    REQUIRE_EQ(P_cmp[7], P[7]);
    REQUIRE_EQ(P_cmp[8], P[8]);
}

TEST_CASE("conv2d_large_test")
{
    int height = 1024;
    int width = 512;
    float* N = new float[height * width];
    float* F = new float[5 * 5];
    float* P = new float[height * width];
    float* P_cmp = new float[height * width];

    std::iota(F, F + 25, 1);
#pragma omp parallel for
    for (int i {}; i < height; i++) {
        for (int j {}; j < width; j++) {
            N[i * width + j] = j % (i + 2);
        }
    }

    conv2d(N, F, P, 2, width, height);
    conv2dOpenmp(N, F, P_cmp, 2, width, height);

#pragma omp parallel for
    for (int i {}; i < height; i++) {
        for (int j {}; j < width; j++) {
            REQUIRE_EQ(P_cmp[i * width + j], P[i * width + j]);
        }
    }

    delete[] N;
    delete[] F;
    delete[] P;
    delete[] P_cmp;
}

void conv2dOpenmp(float* N, float* F, float* P, int radius, int width, int height)
{
#pragma omp parallel for
    for (int i {}; i < height; i++) {
        for (int j {}; j < width; j++) {
            float val {};
            for (int fi {}; fi < 2 * radius + 1; fi++) {
                for (int fj {}; fj < 2 * radius + 1; fj++) {
                    int ni = i - radius + fi;
                    int nj = j - radius + fj;

                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        val += F[fi * radius + fj] * N[ni * width + nj];
                    }
                }
            }
            P[i * width + j] = val;
        }
    }
}