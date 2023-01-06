//
// Created by Nathan on 2022-12-11.
//

#ifndef DEV_QHC_CPP_PROJECTS_RTWEEKEND_H
#define DEV_QHC_CPP_PROJECTS_RTWEEKEND_H
#include <curand_kernel.h>
#include <limits>
// Usings

using std::sqrt;

// Constants

__device__ const double infinity = std::numeric_limits<double>::infinity();
__device__ const double pi = 3.1415926535897932385;

// Utility Functions

__device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}


__device__ inline double random_double() {
    curandState_t state{};
    curand_init(clock(), 0, 0, &state);
    return curand_uniform(&state);
}

__device__ inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers

#include "Ray.h"
#include "Vec3.h"
#endif //DEV_QHC_CPP_PROJECTS_RTWEEKEND_H
