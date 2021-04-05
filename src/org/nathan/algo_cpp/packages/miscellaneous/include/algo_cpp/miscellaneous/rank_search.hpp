//
// Created by Nathan on 2021/4/4.
//

#ifndef CPP_ALL_IN_ONE_RANK_SEARCH_CPP
#define CPP_ALL_IN_ONE_RANK_SEARCH_CPP

#include <vector>
#include <random>
#include <iostream>
#include <exception>

namespace org::nathan::algo_cpp::miscellaneous
{
    using RandEngine_t = std::mt19937;
    using std::vector;
    using std::cout, std::endl;

    class RankSearch final
    {
    private:
        template<typename Number>
        static int randPartition(vector<Number> &a, int start, int end, RandEngine_t &engine)
        {
            std::uniform_int_distribution dist{start, end - 1};
            int pivot_idx{dist(engine)};

            auto pivot{a[pivot_idx]};

            auto temp{a[end - 1]};
            a[end - 1] = pivot;
            a[pivot_idx] = temp;

            int i{start - 1};
            for (int j{start}; j < end - 1; j++)
            {
                if (a[j] <= pivot)
                {
                    auto t{a[j]};
                    a[j] = a[++i];
                    a[i] = t;
                }
            }
            a[end - 1] = a[++i];
            a[i] = pivot;
            return i; //pivot idx
        }

        // select ith smallest element in array
        template<typename Number>
        static Number rankSearch(vector<Number> &a, int start, int end, int ith, RandEngine_t &engine)
        {
            if ((start - end) == 1)
            {
                return a[start];
            }
            int pivot_idx{randPartition(a, start, end, engine)};
            int left_total{pivot_idx - start};
            if (ith == left_total)
            {
                return a[pivot_idx];
            }
            else if (ith < left_total + 1)
            {
                return rankSearch(a, start, pivot_idx, ith, engine);
            }
            else
            {
                return rankSearch(a, pivot_idx + 1, end, ith - left_total - 1, engine);
            }
        }


    public:
        template<typename Number>
        [[maybe_unused]] static Number find(vector<Number> &a, int ith)
        {
            if (a.size() == 0)
            {
                throw std::range_error{"size is zero."};
            }
            std::random_device seed{};
            RandEngine_t engine{seed()};
            return rankSearch(a, 0, a.size(), ith, engine);
        }
    };
}


#endif //CPP_ALL_IN_ONE_RANK_SEARCH_CPP
