//
// Created by Nathan on 2021/3/25.
//

#include "include/lib_central/numerics.h"

auto org::qhc::lib_central::sieveOfEratosthenes(int limit) -> std::vector<int> {
    std::vector<bool> prime{};
    prime.reserve(limit + 1);
    prime.assign(limit + 1, true);
    for (int p = 2; p * p <= limit; p++) {
        if (prime.at(p)) {
            for (int i = p * 2; i <= limit; i += p) {
                prime.at(i) = false;
            }
        }
    }
    std::vector<int> primeNumbers{};
    for (int i = 2; i <= limit; i++) {
        if (prime.at(i)) {
            primeNumbers.push_back(i);
        }
    }
    return std::move(primeNumbers);
}