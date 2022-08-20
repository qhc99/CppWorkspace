//
// Created by Nathan on 2021/4/4.
//
#define BOOST_TEST_MODULE RankSearchTest

#include "lib_central/rank_search.hpp"
#include "lib_central/utils.h"
#include <algorithm>
#include <boost/test/unit_test.hpp>

using org::qhc::lib_central::RankSearch;
using org::qhc::lib_central::shuffledRange;

BOOST_AUTO_TEST_SUITE(RankSearchAPITest) // NOLINT

  BOOST_AUTO_TEST_CASE(rank_search) {
    for (int i = 0; i < 15; i++) {
      std::vector<int> data{shuffledRange(1, 20)};
      int r{RankSearch::find(data, i)};
      std::sort(data.begin(), data.end());
      BOOST_CHECK_EQUAL(data.at(i), r);
    }
  }

BOOST_AUTO_TEST_SUITE_END() // NOLINT