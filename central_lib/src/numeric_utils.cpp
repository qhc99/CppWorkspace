//
// Created by QC on 2021/3/25.
//

#include "lib_central/numeric_utils.h"

auto dev::qhc::Euler::sieveOfEratosthenes(int limit) -> std::vector<int> {
    std::vector<bool> prime{};
    prime.reserve(limit + 1);
    prime.assign(limit + 1, true);
    for(int i = 0; i < limit + 1; ++i){
        prime[i] = true;
    }
    for (int p = 2; p * p <= limit; p++) {
        if (prime[p]) {
            for (int i = p * 2; i <= limit; i += p) {
                prime[i] = false;
            }
        }
    }
    std::vector<int> primeNumbers{};
    for (int i = 2; i <= limit; i++) {
        if (prime[i]) {
            primeNumbers.push_back(i);
        }
    }
    return primeNumbers;
}