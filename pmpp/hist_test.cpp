#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "hist.h"

void hist_test(unsigned int length, unsigned int *out, unsigned int bin_width=2)
{
    auto data = std::make_unique<std::vector<char>>(length);
#pragma omp parallel for
    for (unsigned int i {}; i < length; i++) {
        (*data)[i] = static_cast<char>(i % 26) + 'a';
    }
    hist(data->data(), length, out, bin_width);
}

TEST_CASE("hist_large_test")
{
    constexpr unsigned int length = 26*512;
    auto out{std::make_unique<std::vector<unsigned int>>(13)};
    hist_test(length,out->data());
    for (unsigned int i {}; i < 13; i++) {
        REQUIRE_EQ(1024, (*out)[i]);
    }
}

TEST_CASE("hist_small_test")
{
    constexpr unsigned int length = 26;
    auto out{std::make_unique<std::vector<unsigned int>>(13)};
    hist_test(length,out->data());
        for (unsigned int i {}; i < 13; i++) {
        REQUIRE_EQ(2, (*out)[i]);
    }
}