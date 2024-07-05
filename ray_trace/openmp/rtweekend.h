//
// Created by QC on 2022-12-11.
//

#ifndef DEV_QHC_CPP_PROJECTS_RTWEEKEND_H
#define DEV_QHC_CPP_PROJECTS_RTWEEKEND_H
#include <cmath>
#include <limits>
#include <memory>
// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

double random_double();

inline double random_double(double min, double max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

inline double clamp(double x, double min, double max)
{
    if (x < min) {
        return min;
    }
    if (x > max) {
        return max;
    }
    return x;
}

#endif // DEV_QHC_CPP_PROJECTS_RTWEEKEND_H
