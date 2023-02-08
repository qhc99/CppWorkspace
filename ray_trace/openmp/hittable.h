//
// Created by QC on 2022-12-10.
//

#ifndef DEV_QHC_CPP_PROJECTS_HITTABLE_H
#define DEV_QHC_CPP_PROJECTS_HITTABLE_H
#include "rtweekend.h"
#include "ray.h"

class Material;

struct HitRecord {
    Point3 p;
    Vec3 normal;
    double t;
    bool front_face;
    shared_ptr<Material> mat_ptr;

    inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class Hittable {
public:
    virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const = 0;
};
#endif //DEV_QHC_CPP_PROJECTS_HITTABLE_H
