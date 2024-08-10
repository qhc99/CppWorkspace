#ifndef LEARNCPP_TOOLS_H
#define LEARNCPP_TOOLS_H

#include <vector>
namespace dev::qhc::Euler {
using std::vector;

/**
 * first limit primes
 * @param limit
 * @return
 */
vector<size_t> sieveOfEratosthenes(size_t limit);
} // namespace dev::qhc::Euler

#endif // LEARNCPP_TOOLS_H
