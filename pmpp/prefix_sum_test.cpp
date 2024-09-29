#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "prefix_sum.h"
#include <doctest/doctest.h>

TEST_CASE("prefix_sum_kogge_stone_small_test")
{
    std::vector<float> v(200, 1.F);
    std::vector<float> out(200, 0.F);
    KoggeStoneSegmentScan(v.data(), out.data(), static_cast<unsigned int>(v.size()));
    for (size_t i {}; i < out.size(); ++i) {
        REQUIRE_EQ(static_cast<float>(i + 1), out.at(i));
    }
}

TEST_CASE("prefix_sum_kogge_stone_large_test")
{
    std::vector<float> v(2000, 1.F);
    std::vector<float> out(2000, 0.F);
    KoggeStoneSegmentScan(v.data(), out.data(), static_cast<unsigned int>(v.size()));
    for (size_t i {}; i < out.size(); ++i) {
        int t = i % 512;
        REQUIRE_EQ(static_cast<float>(t + 1), out.at(i));
    }
}

TEST_CASE("prefix_sum_brent_kung_small_test")
{
    std::vector<float> v(200, 1.F);
    std::vector<float> out(200, 0.F);
    BrentKungSegmentScan(v.data(), out.data(), static_cast<unsigned int>(v.size()));
    for (size_t i {}; i < out.size(); ++i) {
        REQUIRE_EQ(static_cast<float>(i + 1), out.at(i));
    }
}

TEST_CASE("prefix_sum_brent_kung_large_test")
{
    std::vector<float> v(2000, 1.F);
    std::vector<float> out(2000, 0.F);
    BrentKungSegmentScan(v.data(), out.data(), static_cast<unsigned int>(v.size()));
    for (size_t i {}; i < out.size(); ++i) {
        int t = i % 1024;
        REQUIRE_EQ(static_cast<float>(t + 1), out.at(i));
    }
}