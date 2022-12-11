//
// Created by Nathan on 2021/3/15.
//

#ifndef ORG_NATHAN_CPP_ALL_IN_ONE_UTILS_H
#define ORG_NATHAN_CPP_ALL_IN_ONE_UTILS_H

#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>


namespace dev::qhc::utils {

    using std::chrono::steady_clock;
    using std::chrono::time_point;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    using std::chrono::seconds;
    using std::chrono::duration_cast;
    using std::vector;
    using steady_clock_time_point = decltype(steady_clock::now());

    /**
     *
     * @return steady_clock::now()
     */
    inline steady_clock_time_point current_time_point() noexcept {
        return steady_clock::now();
    }


    /**
     *
     * @param current current steady_clock::now()
     * @param before before steady_clock::now()
     * @return milliseconds in double
     */
    inline double time_point_duration_to_ms(steady_clock_time_point current, steady_clock_time_point before) {
        if (current < before) {
            throw std::logic_error{"before > current"};
        }
        return duration_cast<milliseconds>(current - before).count();
    }


    /**
     * [low, high) shuffled vector
     * @tparam Number
     * @param low include
     * @param high exclude
     * @return
     */
    auto shuffledRange(int low, int high) -> vector<int>;

    decltype(std::mt19937{std::random_device{}()}) default_rand_engine();

}


#endif //ORG_NATHAN_CPP_ALL_IN_ONE_UTILS_H
