#include <limits>
#include <memory>
#include <vector>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "reduce.h"
#include <doctest/doctest.h>

TEST_CASE("reduce_small_test")
{
    int a1[6] = { -1, -2, -3, -4, -5, -6 };
    int a2[6] = { 1, 2, 3, 4, 5, 6 };
    float f[5] = { -1, -1, -1, -1, -1 };
    int out_i1 { std::numeric_limits<int>::lowest() };
    int out_i2 { std::numeric_limits<int>::max() };
    float out_f {};

    reduce_max_i(a1, 6, &out_i1);
    REQUIRE_EQ(-1, out_i1);

    reduce_min_i(a2, 6, &out_i2);
    REQUIRE_EQ(1, out_i2);

    reduce_add_f(f, 5, &out_f);
    REQUIRE_EQ(-5, out_f);
}

TEST_CASE("reduce_large_test")
{
    size_t length = 1024 * 16;
    auto a1 { std::make_unique<std::vector<int>>(length, 0) };
    auto a2 { std::make_unique<std::vector<int>>(length, 0) };
    auto f { std::make_unique<std::vector<float>>(length, -1.F) };

    int out_i1 { std::numeric_limits<int>::lowest() };
    int out_i2 { std::numeric_limits<int>::max() };
    float out_f {};

    int start { -1 };
    std::generate(a1->begin(), a1->end(), [&]() { return start--; });
    std::iota(a2->data(), a2->data() + a2->size(), 1);

    reduce_max_i(a1->data(), a1->size(), &out_i1);
    REQUIRE_EQ(-1, out_i1);

    reduce_min_i(a2->data(), a2->size(), &out_i2);
    REQUIRE_EQ(1, out_i2);

    reduce_add_f(f->data(), f->size(), &out_f);
    REQUIRE_EQ(-static_cast<float>(f->size()), out_f);
}