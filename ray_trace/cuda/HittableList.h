//
// Created by Nathan on 2022-12-11.
//

#ifndef DEV_QHC_CPP_PROJECTS_HITTABLELIST_H
#define DEV_QHC_CPP_PROJECTS_HITTABLELIST_H

#include "Hittable.h"


class HittableList : public Hittable {
private:
    int idx{0};
    int len{0};
public:
    __device__ explicit HittableList(int max_len) {
        len = max_len;
        objects = new Hittable * [max_len];
    }

    __device__ void add(Hittable *object) {
        if (idx < len) {
            objects[idx] = object;
            idx++;
        }
    }

    __device__ bool hit(
        const Ray &r, double t_min, double t_max, HitRecord &rec) const override;

    __device__ ~HittableList() override{
        for(int i =0; i < idx; i++){
            delete objects[i];
        }
        delete[] objects;
    }
public:
    Hittable **objects{};
};

__device__ inline bool HittableList::hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < idx; i++) {
        auto &object = *objects[i];

        if (object.hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif //DEV_QHC_CPP_PROJECTS_HITTABLELIST_H
