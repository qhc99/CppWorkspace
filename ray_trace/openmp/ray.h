//
// Created by QC on 2022-12-06.
//

#ifndef DEV_QHC_CPP_PROJECTS_RAY_H
#define DEV_QHC_CPP_PROJECTS_RAY_H
#include "vec3.h"

class Ray {
public:
    Ray() = default;
    Ray(const Point3& origin, const Vec3& direction)
        : orig(origin)
        , dir(direction)
    {
    }

    [[nodiscard]] Point3 origin() const { return orig; }
    [[nodiscard]] Vec3 direction() const { return dir; }

    [[nodiscard]] Point3 at(double t) const
    {
        return orig + t * dir;
    }

    Point3 orig;
    Vec3 dir;
};
#endif // DEV_QHC_CPP_PROJECTS_RAY_H
