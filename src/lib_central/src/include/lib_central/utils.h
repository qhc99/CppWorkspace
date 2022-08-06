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


namespace org::qhc::lib_central
{

    using std::chrono::steady_clock;
    using std::chrono::time_point;
    using std::chrono::duration;
    using std::vector;

    /**
     *
     * @return steady_clock::now()
     */
    auto currentTime() -> decltype(steady_clock::now());


  /**
   *
   * @param current current steady_clock::now()
   * @param before before steady_clock::now()
   * @return milliseconds in double
   */
  auto timeIntervalToMilli(
    time_point<steady_clock, duration<double>> current,
    time_point<steady_clock, duration<double>> before) -> double;


  /**
   * [low, high) shuffled vector
   * @tparam Number
   * @param low include
   * @param high exclude
   * @return
   */
  auto shuffledRange(int low, int high) -> vector<int>;
}


#endif //ORG_NATHAN_CPP_ALL_IN_ONE_UTILS_H
