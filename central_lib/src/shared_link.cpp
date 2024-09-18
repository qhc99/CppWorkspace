#include <fmt/core.h>
#include "lib_central/utils.h"

int main()
{
    fmt::print("Hello World!\n");
    auto v {dev::qhc::utils::shuffledRange(0, 6)};
    fmt::print("{}", fmt::join(v, ", "));
    return 0;
}