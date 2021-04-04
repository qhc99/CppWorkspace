//
// Created by Nathan on 2021/3/25.
//

#include "euler/tools.h"

namespace org::nathan::Euler
{

    /**
     * first n primes
     * @param n
     * @return
     */
    std::vector<int> sieveOfEratosthenes(int n)
    {
        std::vector<bool> prime{};
        prime.reserve(n + 1);
        for (auto &&iter : prime)
        {
            iter = true;
        }
        for (int p = 2; p * p <= n; p++)
        {
            if (prime[p])
            {
                for (int i = p * 2; i <= n; i += p)
                {
                    prime[i] = false;
                }
            }
        }
        std::vector<int> primeNumbers{};
        for (int i = 2; i <= n; i++)
        {
            if (prime[i])
            {
                primeNumbers.push_back(i);
            }
        }
        return std::move(primeNumbers);
    }
}