

#include "euler/numerics.h"
#include "utils/utils.h"
#include "algo_cpp/structures/DisjointSet.h"
#include <iostream>
#include <vector>
#include <string>
#include <any>
#include <random>
#include <functional>


using std::cout, std::endl, std::cin, std::string;

using org::nathan::utils::currentTime, org::nathan::utils::timeIntervalToMilli;

using org::nathan::Euler::sieveOfEratosthenes;

using org::nathan::algo_cpp::structures::DisjointSet;

using std::vector;


template<typename T0, typename... T>
void printf2(T0 t0, T... t)
{
    std::cout << t0 << std::endl;
    if constexpr (sizeof...(t) > 0) printf2(t...);
    cout << t0 << endl;
}

double add(int a, double b)
{
    return a + b;
}


int main()
{
    auto add = [](auto &&PH1, auto &&PH2)
    { return ::add(std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH1)>(PH1)); };

    cout << add(2.0, 1) << endl;
    return 0;
}
