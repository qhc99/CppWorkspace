//
// Created by Nathan on 2021/3/15.
//
#include "lib_central/utils.h"

namespace dev::qhc::utils {

    decltype(steady_clock::now()) current_time_point() {
        return steady_clock::now();
    }

    double time_point_interval_to_ms(
            time_point<steady_clock, duration<double>> current,
            time_point<steady_clock, duration<double>> before) {
        if (before > current) {
            throw std::logic_error{"before > current"};
        }
        return static_cast<duration<double>>(current - before).count() * 1000;
    }

    vector<int> shuffledRange(int low, int high) {
        if (high <= low) {
            throw std::logic_error("low >= high");
        }
        vector<int> r(high - low);
        std::generate(r.begin(), r.end(), [n = 0]() mutable { return n++; });
        std::shuffle(r.begin(), r.end(), std::mt19937{std::random_device{}()});
        return std::move(r);
    }

    decltype(std::mt19937{ std::random_device{}() }) default_rand_engine()
    {
        return std::mt19937{ std::random_device{}() };
    }
}
