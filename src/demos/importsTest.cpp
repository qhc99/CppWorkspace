//
// Created by Nathan on 2021/4/15.
//
#include "demos/importsTest.h"


void importsTest()
{
    DisjointSet a{};
    vector<int> v{1, 2, 3};
    cout << RankSearch::find(v, 2) << endl;
    auto t1 = currentTime();
    auto temp{sieveOfEratosthenes(1000)};
    cout << temp.at(10) << endl;
    auto t2 = currentTime();
    cout << "time: " << timeIntervalToMilli(t2, t1) << "ms" << endl;
    try
    {
        org::nathan::utils::miscellaneous::WriteBGFile();
    }
    catch(const std::exception &e)
    {
        cout << e.what() << endl;
    }
}