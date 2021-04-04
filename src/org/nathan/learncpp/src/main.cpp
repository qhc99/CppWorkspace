

#include "euler/tools.h"
#include "utils/utils.h"
#include "algo_cpp/structures/DisjointSet.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>


using std::cout, std::endl, std::cin, std::string;

using org::nathan::utils::currentTime, org::nathan::utils::timeIntervalToMilli;

using org::nathan::Euler::sieveOfEratosthenes;

using org::nathan::algo_cpp::structures::DisjointSet;


int main()
{

    std::random_device seed{};
    std::mt19937 engine{seed()};
    auto &e{engine};
    std::uniform_int_distribution dist{1, 100 - 1};
    for (int i = 0; i < 10; i++)
    {
        cout << dist(e) << endl;
    }
    return 0;
}
