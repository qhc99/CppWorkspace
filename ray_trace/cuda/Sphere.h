//
// Created by Nathan on 2022-12-10.
//

#ifndef DEV_QHC_CPP_PROJECTS_SPHERE_H
#define DEV_QHC_CPP_PROJECTS_SPHERE_H

#include "Hittable.h"
#include "Material.h"
#include "Vec3.h"

class Sphere : public Hittable {
public:
    __device__ Sphere() {};

    __device__ Sphere(Point3 cen, double r, Material *m)
        : center(cen), radius(r), mat_ptr(m) {};

    __device__ virtual bool hit(
        const Ray &r, double t_min, double t_max, HitRecord &rec) const;

public:
    Point3 center;
    double radius{};
    Material *mat_ptr{};

    __device__ ~Sphere() override {
        delete mat_ptr;
    }
};


__device__ bool Sphere::hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const {
    Vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    rec.normal = (rec.p - center) / radius;
    Vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif //DEV_QHC_CPP_PROJECTS_SPHERE_H