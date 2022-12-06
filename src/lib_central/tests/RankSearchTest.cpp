//
// Created by Nathan on 2021/4/4.
//
#define BOOST_TEST_MODULE RankSearchTest

#include "lib_central/rank_search.hpp"
#include "lib_central/utils.h"
#include <algorithm>
#include <boost/test/unit_test.hpp>

using dev::qhc::lib_central::RankSearch;
using dev::qhc::utils::shuffledRange;

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