//
// Created by QC on 2021/4/4.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "lib_central/rank_search.hpp"
#include "lib_central/utils.h"
#include "lib_central/doctest.h"
#include <algorithm>

using dev::qhc::lib_central::RankSearch;
using dev::qhc::utils::shuffledRange;

TEST_CASE("RankSearchAPITest.rank_search") {
    for (int i = 0; i < 15; i++) {
        std::vector<int> data{shuffledRange(1, 20)};
        int r{RankSearch::find(data, 10)};
        std::sort(data.begin(), data.end());
        REQUIRE(data[10] == r);
    }
}

