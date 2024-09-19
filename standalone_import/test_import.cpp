#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "lib_central/dll.h"

TEST_CASE("shared_lib_link_test")
{
    auto v {dev::qhc::utils::shuffledRange(0, 6)};
    REQUIRE(v[0] < 100);
}
