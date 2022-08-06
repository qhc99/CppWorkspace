//
// Created by Nathan on 2021/4/4.
//
#define BOOST_TEST_MODULE RankSearchTest

#include "algo_cpp/rank_search.hpp"
#include "utils/utils.h"
#include <algorithm>
#include <boost/test/unit_test.hpp>

using org::nathan::algo_cpp::RankSearch;
using org::nathan::utils::shuffledRange;

BOOST_AUTO_TEST_SUITE(RankSearchAPITest) // NOLINT

BOOST_AUTO_TEST_CASE(rank_search) // NOLINT
        {
                for (int i = 0; i < 15; i++)
                {
                    std::vector<int> data{std::move(shuffledRange(1, 20))};
                    int r{RankSearch::find(data, 10)};
                    std::sort(data.begin(), data.end());
                    BOOST_CHECK_EQUAL(data[10], r);
                }
        }

BOOST_AUTO_TEST_SUITE_END() // NOLINT