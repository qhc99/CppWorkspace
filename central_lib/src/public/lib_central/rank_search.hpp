//
// Created by QC on 2021/4/4.
//

#ifndef ORG_QC_CPP_CENTRAL_LIB_RANK_SEARCH_CPP
#define ORG_QC_CPP_CENTRAL_LIB_RANK_SEARCH_CPP

#include <random>
#include <stdexcept>
#include <vector>

#include "lib_central/concepts_utils.h"

namespace dev::qhc::central_lib::rank_search {
using RandEngine_t = std::mt19937;
using std::vector;

namespace {
    template <concepts_utils::Comparable T>
    int randPartition(vector<T>& a, int start, int end,
        RandEngine_t& engine)
    {
        std::uniform_int_distribution<int> dist { start, end - 1 };
        int pivot_idx { dist(engine) };

        auto pivot { a.at(pivot_idx) };

        auto& temp { a.at(end - 1) };
        a.at(end - 1) = pivot;
        a.at(pivot_idx) = temp;

        int i { start - 1 };
        for (int j { start }; j < end - 1; j++) {
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
    template <typename T>
    T rankSearch(vector<T>& a, int start, int end, int ith,
        RandEngine_t& engine)
    {
        if ((start - end) == 1) {
            return a.at(start);
        }
        int pivot_idx { randPartition(a, start, end, engine) };
        int left_total { pivot_idx - start };
        if (ith == left_total) {
            return a.at(pivot_idx);
        } else if (ith < left_total + 1) {
            return rankSearch(a, start, pivot_idx, ith, engine);
        } else {
            return rankSearch(a, pivot_idx + 1, end, ith - left_total - 1, engine);
        }
    }
} // namespace

template <typename T>
static T find(vector<T>& a, int ith)
{
    if (a.empty()) {
        throw std::logic_error { "size is zero." };
    }
    std::random_device seed {};
    RandEngine_t engine { seed() };
    return rankSearch(a, 0, a.size(), ith, engine);
}

} // namespace dev::qhc::central_lib::rank_search

#endif // ORG_QC_CPP_CENTRAL_LIB_RANK_SEARCH_CPP
