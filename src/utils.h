//
// Created by Nathan on 2021/3/15.
//

#ifndef LEARNCPP_UTILS_H
#define LEARNCPP_UTILS_H

#include <chrono>


namespace org::nathan::utils {

    using namespace std::chrono;

    /**
     *
     * @return steady_clock::now()
     */
    [[maybe_unused]] typeof(steady_clock::now()) currentTime();


    /**
     *
     * @param later later steady_clock::now()
     * @param former former steady_clock::now()
     * @return milliseconds in double
     */
    [[maybe_unused]] double intervalToMilli(
            time_point<steady_clock, duration<double>> later,
            time_point<steady_clock, duration<double>> former);
}


#endif //LEARNCPP_UTILS_H
