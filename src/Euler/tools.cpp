//
// Created by Nathan on 2021/3/25.
//

#include <vector>

namespace org::nathan::Euler
{
    std::vector<long long> SieveOfAtkin(int limit)
    {

        std::vector<long long> res{};
        res.reserve(32);

        if (limit > 2)
        {
            res.push_back(2L);
        }

        if (limit > 3)
        {
            res.push_back(3L);
        }

        bool *sieve = new bool[limit];

        for (int i = 0; i < limit; i++)
            sieve[i] = false;

        for (int x = 1; x * x < limit; x++)
        {
            for (int y = 1; y * y < limit; y++)
            {

                int n = (4 * x * x) + (y * y);
                if (n <= limit && (n % 12 == 1 || n % 12 == 5))
                {
                    sieve[n] ^= true;
                }

                n = (3 * x * x) + (y * y);
                if (n <= limit && n % 12 == 7)
                {
                    sieve[n] ^= true;
                }

                n = (3 * x * x) - (y * y);
                if (x > y && n <= limit && n % 12 == 11)
                {
                    sieve[n] ^= true;
                }
            }
        }

        for (int r = 5; r * r < limit; r++)
        {
            if (sieve[r])
            {
                for (int i = r * r; i < limit;
                     i += r * r)
                    sieve[i] = false;
            }
        }

        for (int a = 5; a < limit; a++)
        {
            if (sieve[a])
            {
                res.push_back(a);
            }
        }

        delete[] sieve;

        return std::move(res);
    }
}