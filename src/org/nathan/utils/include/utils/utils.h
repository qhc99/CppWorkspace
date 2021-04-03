//
// Created by Nathan on 2021/3/15.
//

#ifndef ORG_NATHAN_LEARNCPP_UTILS_H
#define ORG_NATHAN_LEARNCPP_UTILS_H

#include <chrono>


namespace org::nathan::utils
{

    using std::chrono::steady_clock;
    using std::chrono::time_point;
    using std::chrono::duration;

    /**
     *
     * @return steady_clock::now()
     */
    [[maybe_unused]] typeof(steady_clock::now()) currentTime();


    /**
     *
     * @param current current steady_clock::now()
     * @param last last steady_clock::now()
     * @return milliseconds in double
     */
    [[maybe_unused]] double timeIntervalToMilli(
            time_point<steady_clock, duration<double>> current,
            time_point<steady_clock, duration<double>> last);
}


#endif //ORG_NATHAN_LEARNCPP_UTILS_H
