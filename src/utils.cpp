//
// Created by Nathan on 2021/3/15.
//
#include <chrono>

namespace org::nathan::utils {

    using namespace std::chrono;

    [[maybe_unused]] typeof(steady_clock::now()) currentTime() {
        return steady_clock::now();
    }

    [[maybe_unused]] double intervalToMilli(
            time_point<steady_clock, duration<double>> later,
            time_point<steady_clock, duration<double>> former) {
        return static_cast<duration<double>>(later - former).count() * 1000;
    }
}
