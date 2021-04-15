//
// Created by Nathan on 2021/4/4.
//

#include "algo_cpp/miscellaneous/rank_search.hpp"
#include "utils/utils.h"
#include <gtest/gtest.h>
#include <algorithm>


using org::nathan::algo_cpp::miscellaneous::RankSearch;
using org::nathan::utils::shuffledRange;

TEST(rankSearch, test)// NOLINT
{
    for (int i = 0; i < 100; i++)
    {
        std::vector<int> data{std::move(shuffledRange(1, 20))};
        int r{RankSearch::find(data, 10)};
        std::sort(data.begin(), data.end());
        EXPECT_EQ(data[10], r);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}