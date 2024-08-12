#include "lib_central/numeric_utils.h"

std::vector<size_t> dev::qhc::Euler::sieveOfEratosthenes(size_t limit)
{
    std::vector<bool> prime {};
    prime.reserve(limit + 1);
    prime.assign(limit + 1, true);
    for (size_t p = 2; p * p <= limit; p++) {
        if (prime.at(p)) {
            for (size_t i = p * 2; i <= limit; i += p) {
                prime.at(i) = false;
            }
        }
    }
    std::vector<size_t> primeNumbers {};
    for (size_t i = 2; i <= limit; i++) {
        if (prime.at(i)) {
            primeNumbers.push_back(i);
        }
    }
    return primeNumbers;
}