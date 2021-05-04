//
// Created by Nathan on 2021/3/25.
//

#include "euler/numerics.h"

std::vector<int> org::nathan::Euler::sieveOfEratosthenes(int n)
{
    std::vector<bool> prime{};
    prime.reserve(n + 1);
    prime.assign(n + 1, true);
    for (int p = 2; p * p <= n; p++)
    {
        if(prime.at(p))
        {
            for(int i = p * 2; i <= n; i += p)
            {
                prime.at(i) = false;
            }
        }
    }
    std::vector<int> primeNumbers{};
    for (int i = 2; i <= n; i++)
    {
        if(prime.at(i))
        {
            primeNumbers.push_back(i);
        }
    }
    return std::move(primeNumbers);
}