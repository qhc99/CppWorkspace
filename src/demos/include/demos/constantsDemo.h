//
// Created by Nathan on 2021/3/16.
//

#ifndef LEARNCPP_CONSTANTSDEMO_H
#define LEARNCPP_CONSTANTSDEMO_H

#include <random>
#include <iostream>
// use inline constexpr
namespace constantsDemo
{
    inline constexpr double pi{3.14159}; // note: now inline constexpr
    inline constexpr double avogadro{6.0221413e23};
    inline constexpr double my_gravity{9.2}; // m/s^2 -- gravity is light on this planet
}

#endif //LEARNCPP_CONSTANTSDEMO_H
