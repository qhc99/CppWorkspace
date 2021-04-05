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

    [[maybe_unused]] vector<int> shuffledRange(int low, int high)
    {
        if (high <= low)
        {
            throw std::logic_error("low >= high");
        }
        vector<int> r(high - low);
        std::generate(r.begin(), r.end(), [n = 0]() mutable
        { return n++; });
        std::shuffle(r.begin(), r.end(), std::mt19937{std::random_device{}()});
        return std::move(r);
    }
}
