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
    t[2] = 2; // NOLINT cppcoreguidelines-pro-bounds-pointer-arithmetic
    delete[] t;
}

void use_after_free()
{
    int* t = new int[2];
    delete[] t;
    t[1] = 1; // NOLINT cppcoreguidelines-pro-bounds-pointer-arithmetic
}
}
