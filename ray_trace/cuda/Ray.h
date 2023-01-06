//
// Created by Nathan on 2022-12-06.
//

#ifndef DEV_QHC_CPP_PROJECTS_RAY_H
#define DEV_QHC_CPP_PROJECTS_RAY_H
#include "Vec3.h"

class Ray {
public:
    __device__ Ray() {};
    __device__ Ray(const Point3& origin, const Vec3& direction)
        : orig(origin), dir(direction)
    {}

    __device__ Point3 origin() const  { return orig; }
    __device__ Vec3 direction() const { return dir; }

    __device__ Point3 at(double t) const {
        return orig + t*dir;
    }

public:
    Point3 orig;
    Vec3 dir;
};
#endif //DEV_QHC_CPP_PROJECTS_RAY_H
