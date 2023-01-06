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
    __device__ static void init(HittableList* self,int max_len){
        self->len = max_len;
        self->objects = new Hittable *[max_len];
        for (int i = 0; i < self->len; i++) {
            self->objects[i] = nullptr;
        }
    }

    __device__ void add(Hittable *object) {
        if (idx < len) {
            objects[idx] = object;
            idx++;
        }
    }

//    __device__ HittableList& operator=(const HittableList& other){
//        objects = other.objects;
//    };

    __device__ bool hit(
        const Ray &r, double t_min, double t_max, HitRecord &rec) const override;

public:
    Hittable **objects{};
};

__device__ bool HittableList::hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0;i < len; i++) {
        auto object = objects[i];

        if (object != nullptr && object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif //DEV_QHC_CPP_PROJECTS_HITTABLELIST_H
