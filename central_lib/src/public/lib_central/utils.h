#ifndef ORG_QC_CPP_ALL_IN_ONE_UTILS_H
#define ORG_QC_CPP_ALL_IN_ONE_UTILS_H

#include <chrono>
#include <random>
#include <stdexcept>
#include <vector>

namespace dev::qhc::utils {

using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::seconds;
using std::chrono::steady_clock;
using std::chrono::time_point;
using steady_clock_time_point = decltype(steady_clock::now());
/**
 *
 * @return steady_clock::now()
 */
inline steady_clock_time_point current_time_point() noexcept
{
    return steady_clock::now();
}

/**
 *
 * @param current current steady_clock::now()
 * @param before before steady_clock::now()
 * @return milliseconds in double
 */
inline long long time_point_duration_to_us(steady_clock_time_point current,
    steady_clock_time_point before)
{
    if (current < before) {
        throw std::logic_error { "before > current" };
    }
    return duration_cast<microseconds>(current - before).count();
}

/**
 * [low, high) shuffled vector
 * @tparam Number
 * @param low public
 * @param high exclude
 * @return
 */
auto shuffledRange(int low, int high) -> vector<int>;

void leak();

void out_of_range_access();

void use_after_free();

inline decltype(std::mt19937 { std::random_device {}() }) default_rand_engine()
{
    return std::mt19937 { std::random_device {}() };
};

} // namespace dev::qhc::utils

#endif // ORG_QC_CPP_ALL_IN_ONE_UTILS_H
