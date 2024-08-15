#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "lib_central/utils.h"

using dev::qhc::utils::out_of_range_access;
TEST_CASE("ASAN_out_of_range_access_test")
{
    out_of_range_access();
}
