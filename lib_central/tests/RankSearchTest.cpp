//
// Created by Nathan on 2021/4/4.
//
#define BOOST_TEST_MODULE RankSearchTest

#include "lib_central/rank_search.hpp"
#include "lib_central/utils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
using dev::qhc::lib_central::RankSearch;
using dev::qhc::utils::shuffledRange;

TEST(RankSearchAPITest, rank_search) {
    for (int i = 0; i < 15; i++) {
        std::vector<int> data{std::move(shuffledRange(1, 20))};
        int r{RankSearch::find(data, 10)};
        std::sort(data.begin(), data.end());
        EXPECT_EQ(data[10], r);
    }
    std::cout << "rank";
}

