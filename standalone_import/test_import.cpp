#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "lib_central/dll.h"
#include <numeric>

TEST_CASE("shared_lib_link_test")
{
    auto v {dev::qhc::utils::shuffledRange(0, 6)};
    auto res{std::accumulate(v.begin(), v.end(), 0)};
    REQUIRE_EQ(15,res);
}
