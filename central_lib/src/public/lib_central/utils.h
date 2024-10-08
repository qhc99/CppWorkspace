#ifndef DEV_QC_CENTRAL_LIB_UTILS_H
#define DEV_QC_CENTRAL_LIB_UTILS_H

#ifdef USE_SHARED
  #if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef BUILDING_CENTRAL_LIB
      #define CENTRAL_LIB_API __declspec(dllexport)
    #else
      #define CENTRAL_LIB_API __declspec(dllimport)
    #endif
  #else
    #ifdef BUILDING_CENTRAL_LIB
      #define CENTRAL_LIB_API __attribute__((visibility("default")))
    #else
      #define CENTRAL_LIB_API
    #endif
  #endif
#else
  #define CENTRAL_LIB_API
#endif

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

CENTRAL_LIB_API vector<int> shuffledRange(int low, int high);

void leak();

void out_of_range_access();

void use_after_free();

inline decltype(std::mt19937 { std::random_device {}() }) default_rand_engine()
{
    return std::mt19937 { std::random_device {}() };
};

} // namespace dev::qhc::utils

#endif // DEV_QC_CPP_ALL_IN_ONE_UTILS_H
