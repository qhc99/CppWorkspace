#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "lib_central/utils.h"

using dev::qhc::utils::use_after_free;

TEST_CASE("ASAN_use_after_free_test")
{
    use_after_free();
}
