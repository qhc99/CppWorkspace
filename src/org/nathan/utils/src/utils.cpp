//
// Created by Nathan on 2021/3/15.
//
#include "utils/utils.h"

namespace org::nathan::utils
{

    [[maybe_unused]] typeof(steady_clock::now()) currentTime()
    {
        return steady_clock::now();
    }

    [[maybe_unused]] double timeIntervalToMilli(
            time_point<steady_clock, duration<double>> current,
            time_point<steady_clock, duration<double>> before)
    {
        if (before > current)
        {
            throw std::logic_error("before > current");
        }
        return static_cast<duration<double>>(current - before).count() * 1000;
    }


}
