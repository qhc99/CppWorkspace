#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "lib_central/utils.h"


using dev::qhc::utils::leak;

// Not supported on windows
TEST_CASE("ASAN_leak_test")
{
    leak();
}
