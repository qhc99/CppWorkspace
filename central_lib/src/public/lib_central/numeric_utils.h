#ifndef DEV_QC_CENTRAL_LIB_NUMERIC_TOOLS_H
#define DEV_QC_CENTRAL_LIB_NUMERIC_TOOLS_H

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
