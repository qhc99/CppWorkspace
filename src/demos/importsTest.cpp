//
// Created by Nathan on 2021/4/15.
//
#include "demos/importsTest.h"


void importsTest()
{
    auto t1 = currentTime();
    DisjointSet a{};
    vector<int> v{1, 2, 3};
    cout << RankSearch::find(v, 2) << endl;
    auto t2 = currentTime();
    auto temp{sieveOfEratosthenes(20)};
    cout << temp[1] << endl;
    cout << timeIntervalToMilli(t2, t1) << endl;
}