#include <vector>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "merge_sort.h"
#include <doctest/doctest.h>

TEST_CASE("merge_sort_small_test")
{
    std::vector<int> a(100);
    std::iota(a.begin(), a.end(), 0);
    std::transform(a.begin(), a.end(), a.begin(), [](int x) { return x * 2; });

    std::vector<int> b(100);
    std::iota(b.begin(), b.end(), 0);
    std::transform(b.begin(), b.end(), b.begin(), [](int x) { return x * 2 + 1; });

    std::vector<int> c(a.size() + b.size());
    mergeTiles(a.data(), static_cast<int>(a.size()), b.data(), static_cast<int>(b.size()), c.data());

    for (size_t i {}; i < c.size(); i++) {
        REQUIRE_EQ(i, c.at(i));
    }
}

TEST_CASE("merge_sort_large_test")
{
    std::vector<int> a(2000);
    std::iota(a.begin(), a.end(), 0);
    std::transform(a.begin(), a.end(), a.begin(), [](int x) { return x * 2; });

    std::vector<int> b(2000);
    std::iota(b.begin(), b.end(), 0);
    std::transform(b.begin(), b.end(), b.begin(), [](int x) { return x * 2 + 1; });

    std::vector<int> c(a.size() + b.size());
    mergeTiles(a.data(), static_cast<int>(a.size()), b.data(), static_cast<int>(b.size()), c.data());

    for (size_t i { 1 }; i < c.size(); i++) {
            REQUIRE(c.at(i) > c.at(i - 1));
    }
}