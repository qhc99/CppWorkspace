//
// Created by Nathan on 2021/3/15.
//

#include "demos/randomDemo.h"

namespace org::nathan::randomDemo
{
    [[maybe_unused]] void randomDemo()
    {
        std::random_device dev;
        std::mt19937 mersenne(dev());
        std::uniform_int_distribution distribution{1, 6};
        for (int count{1}; count <= 48; ++count)
        {
            std::cout << distribution(mersenne) << '\t'; // generate a roll of the distribution here
            if (count % 6 == 0)
                std::cout << '\n';
        }
    }
}