//
// Created by Nathan on 2022-12-10.
//

#ifndef DEV_QHC_CPP_PROJECTS_HITTABLE_H
#define DEV_QHC_CPP_PROJECTS_HITTABLE_H

#include "ray.h"

struct hit_record {
    point3 p;
    vec3 normal;
    double t;
};

class hittable {
public:
    virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};
#endif //DEV_QHC_CPP_PROJECTS_HITTABLE_H
