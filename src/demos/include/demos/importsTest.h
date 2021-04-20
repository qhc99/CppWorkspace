//
// Created by Nathan on 2021/4/15.
//

#ifndef CPP_ALL_IN_ONE_IMPORTSTEST_H
#define CPP_ALL_IN_ONE_IMPORTSTEST_H

#include <iostream>
#include <vector>
#include <string>
#include "euler/numerics.h"
#include "utils.h"
#include "DisjointSet.h"
#include "rank_search.hpp"


using std::cout, std::endl, std::cin, std::string;
using org::nathan::utils::currentTime, org::nathan::utils::timeIntervalToMilli;
using org::nathan::Euler::sieveOfEratosthenes;
using org::nathan::algo_cpp::miscellaneous::RankSearch;
using org::nathan::algo_cpp::structures::DisjointSet;
using std::vector;

void importsTest();

#endif //CPP_ALL_IN_ONE_IMPORTSTEST_H
