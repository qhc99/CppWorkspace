#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "lib_central/numeric_utils.h"

TEST_CASE("numeric_utils.sieveOfEratosthenes")
{
    auto ret { dev::qhc::Euler::sieveOfEratosthenes(20) };
    REQUIRE(ret.at(0) == 2);
    REQUIRE(ret.at(1) == 3);
    REQUIRE(ret.at(2) == 5);
    REQUIRE(ret.at(3) == 7);
    REQUIRE(ret.at(4) == 11);
    REQUIRE(ret.at(5) == 13);
    REQUIRE(ret.at(6) == 17);
    REQUIRE(ret.at(7) == 19);
    REQUIRE(ret.size() == 8);
}
