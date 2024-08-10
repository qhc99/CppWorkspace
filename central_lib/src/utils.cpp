#include "lib_central/utils.h"

namespace dev::qhc::utils {

vector<int> shuffledRange(int low, int high)
{
    if (high <= low) {
        throw std::logic_error("low >= high");
    }
    vector<int> r(high - low);
    std::generate(r.begin(), r.end(), [n = 0]() mutable { return n++; });
    std::shuffle(r.begin(), r.end(), std::mt19937 { std::random_device {}() });
    return r;
}

void leak()
{
    int* t = new int[2];
    t[1] = 1; // NOLINT cppcoreguidelines-pro-bounds-pointer-arithmetic
}

void out_of_range_access()
{
    int* t = new int[2];
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6200 6386)
#endif
    t[2] = 2; // NOLINT cppcoreguidelines-pro-bounds-pointer-arithmetic
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    delete[] t;
}

void use_after_free()
{
    int* t = new int[2];
    delete[] t;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6001)
#endif
    t[1] = 1; // NOLINT cppcoreguidelines-pro-bounds-pointer-arithmetic
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}
}
