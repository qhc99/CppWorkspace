//
// Created by Nathan on 2021/4/4.
//

#include "algo_cpp/miscellaneous/rank_search.hpp"
#include "utils/utils.h"
#include <gtest/gtest.h>
#include <algorithm>


using org::nathan::algo_cpp::miscellaneous::RankSearch;
using org::nathan::utils::shuffledRange;

TEST(rankSearch, test)
{
    for (int i = 0; i < 100; i++)
    {
        std::vector<double> data{std::move(shuffledRange<double>(1, 20))};
        double r{RankSearch::find(data, 2)};
        std::sort(data.begin(), data.end());
        EXPECT_EQ(data[2], r);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}