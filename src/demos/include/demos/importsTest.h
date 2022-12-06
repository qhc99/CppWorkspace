//
// Created by qhc on 2021/4/15.
//

#ifndef CPP_ALL_IN_ONE_IMPORTSTEST_H
#define CPP_ALL_IN_ONE_IMPORTSTEST_H

#include <iostream>
#include <vector>
#include <string>
#include "lib_central/numerics.h"
#include "lib_central/utils.h"
#include "lib_central/DisjointSet.h"
#include "lib_central/rank_search.hpp"


using std::cout, std::endl, std::cin, std::string;
using dev::qhc::utils::currentTime, dev::qhc::utils::timeIntervalToMilli;
using dev::qhc::Euler::sieveOfEratosthenes;
using dev::qhc::lib_central::RankSearch;
using dev::qhc::lib_central::DisjointSet;
using std::vector;

void importsTest();

#endif //CPP_ALL_IN_ONE_IMPORTSTEST_H
