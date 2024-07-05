#include "rtweekend.h"
#include <random>

double random_double()
{

    static thread_local std::mt19937 rng(std::random_device {}());
    static thread_local std::uniform_int_distribution<> dist(0, RAND_MAX);
    return dist(rng) / (RAND_MAX + 1.);
}