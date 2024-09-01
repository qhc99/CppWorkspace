#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "conv2d.h"
#include <doctest/doctest.h>

TEST_CASE("conv2d_small_test")
{

    float N[9];
    float F[9];
    float P[9];

    std::fill(N, N + 9, 1);
    std::fill(F, F + 9, 1);

    conv2d(N,F,P,1,3,3);

    REQUIRE_EQ(4, P[0]);
    REQUIRE_EQ(6, P[1]);
    REQUIRE_EQ(4, P[2]);
    REQUIRE_EQ(6, P[3]);
    REQUIRE_EQ(9, P[4]);
    REQUIRE_EQ(6, P[5]);
    REQUIRE_EQ(4, P[6]);
    REQUIRE_EQ(6, P[7]);
    REQUIRE_EQ(4, P[8]);
}

TEST_CASE("conv2d_large_test")
{
}