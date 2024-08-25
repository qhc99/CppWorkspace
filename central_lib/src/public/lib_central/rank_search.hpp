#ifndef DEV_QC_CENTRAL_LIB_RANK_SEARCH_CPP
#define DEV_QC_CENTRAL_LIB_RANK_SEARCH_CPP

#include <random>
#include <stdexcept>
#include <vector>

#include "lib_central/concepts_utils.h"

namespace dev::qhc::central_lib::rank_search {
using RandEngine_t = std::mt19937;
using std::vector;

namespace {
    template <concepts_utils::Comparable T>
    size_t rand_partition(vector<T>& a, size_t start, size_t end,
        RandEngine_t& engine)
    {
        std::uniform_int_distribution<size_t> dist { start, end - 1 };
        size_t pivot_idx { dist(engine) };

        auto pivot { a.at(pivot_idx) };

        auto& temp { a.at(end - 1) };
        a.at(end - 1) = pivot;
        a.at(pivot_idx) = temp;

        size_t i { start - 1 };
        for (size_t j { start }; j < end - 1; j++) {
            if (a.at(j) <= pivot) {
                auto t { a.at(j) };
                a.at(j) = a.at(++i);
                a.at(i) = t;
            }
        }
        a.at(end - 1) = a.at(++i);
        a.at(i) = pivot;
        return i; // pivot idx
    }

    // select ith smallest element in array
    template <concepts_utils::Comparable T>
    T rank_search(vector<T>& a, size_t start, size_t end, size_t ith,
        RandEngine_t& engine)
    {
        if ((start - end) == 1) {
            return a.at(start);
        }
        size_t pivot_idx { rand_partition(a, start, end, engine) };
        size_t left_total { pivot_idx - start };
        if (ith == left_total) {
            return a.at(pivot_idx);
        } else if (ith < left_total + 1) {
            return rank_search(a, start, pivot_idx, ith, engine);
        } else {
            return rank_search(a, pivot_idx + 1, end, ith - left_total - 1, engine);
        }
    }
} // namespace

template <concepts_utils::Comparable T>
static T find(vector<T>& a, size_t ith)
{
    if (a.empty()) {
        throw std::logic_error { "size is zero." };
    }
    std::random_device seed {};
    RandEngine_t engine { seed() };
    return rank_search(a, 0, a.size(), ith, engine);
}

} // namespace dev::qhc::central_lib::rank_search

#endif // DEV_QC_CPP_CENTRAL_LIB_RANK_SEARCH_CPP
