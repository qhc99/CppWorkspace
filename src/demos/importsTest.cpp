//
// Created by Nathan on 2021/4/15.
//
#include "demos/importsTest.h"


void importsTest() {
    DisjointSet a{};
    DisjointSet b{};
    b.unionSet(a);
    cout << "Disjoint set: " << (&b.findGroupRep() == &a) << endl;
    vector<int> v{1, 2, 3};
    cout << RankSearch::find(v, 2) << endl;
    constexpr int len(1000);
    auto t1 = currentTime();
    auto temp{sieveOfEratosthenes(len)};
    cout << temp.size() << "th: " << temp.at(temp.size() - 1) << endl;
    auto t2 = currentTime();
    cout << "time: " << timeIntervalToMilli(t2, t1) << "ms" << endl;
}