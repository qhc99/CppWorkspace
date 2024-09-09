#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "hist.h"
#include <doctest/doctest.h>

constexpr unsigned int bin_width { 2 };
constexpr unsigned int bin_count { 13 };

void hist_test(unsigned int length, unsigned int* out, bool agg)
{
    auto data = std::make_unique<std::vector<char>>(length);
#pragma omp parallel for
    for (unsigned int i {}; i < length; i++) {
        (*data)[i] = static_cast<char>(i % 26) + 'a';
    }
    hist(data->data(), length, out, bin_width, agg);
}

TEST_CASE("hist_large_test")
{
    constexpr unsigned int length = 26 * 512;
    auto out { std::make_unique<std::vector<unsigned int>>(bin_count) };
    hist_test(length, out->data(), false);
    for (unsigned int i {}; i < bin_count; i++) {
        REQUIRE_EQ(1024, (*out)[i]);
    }
}

TEST_CASE("hist_agg_large_test")
{
    constexpr unsigned int length = 26 * 512;
    auto out2 { std::make_unique<std::vector<unsigned int>>(bin_count) };
    hist_test(length, out2->data(), true);
    for (unsigned int i {}; i < bin_count; i++) {
        REQUIRE_EQ(1024, (*out2)[i]);
    }
}


TEST_CASE("hist_small_test")
{
    constexpr unsigned int length = 26;
    auto out { std::make_unique<std::vector<unsigned int>>(bin_count) };
    hist_test(length, out->data(), false);
    for (unsigned int i {}; i < bin_count; i++) {
        REQUIRE_EQ(2, (*out)[i]);
    }
}

TEST_CASE("hist_agg_small_test")
{
    constexpr unsigned int length = 26;
    auto out2 { std::make_unique<std::vector<unsigned int>>(bin_count) };
    hist_test(length, out2->data(), true);
    for (unsigned int i {}; i < bin_count; i++) {
        REQUIRE_EQ(2, (*out2)[i]);
    }
}