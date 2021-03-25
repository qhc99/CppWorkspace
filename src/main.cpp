
#include <iostream>
#include <vector>
#include "utils/utils.h"
#include "Euler/tools.h"

using std::cout, std::endl;

using namespace org::nathan::utils;

using namespace org::nathan::Euler;


int main()
{
    auto t1 = currentTime();
    int limit = 10000;
    auto primes = SieveOfAtkin(limit);
    long long res = 600851475143L;
    for (long prime : primes)
    {
        while (res % prime == 0)
        {
            res /= prime;
            if (res == 1)
            {
                cout << prime << endl;
                break;
            }
        }
    }
    cout << res << endl;
    auto t2 = currentTime();
    cout << intervalToMilli(t2, t1) << endl;

    return 0;
}
