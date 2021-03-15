//
// Created by Nathan on 2021/3/15.
//

#include <random>
#include <iostream>
namespace org::nathan::randomDemo{
    void randomDemo(){
        std::random_device dev;
        std::mt19937 mersenne( dev() );
        std::uniform_int_distribution die{ 1, 6 };
        for (int count{ 1 }; count <= 48; ++count)
        {
            std::cout << die(mersenne) << '\t'; // generate a roll of the die here
            if (count % 6 == 0)
                std::cout << '\n';
        }
    }
}