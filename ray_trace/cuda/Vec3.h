//
// Created by Nathan on 2022-12-05.
//

#ifndef DEV_QHC_CPP_PROJECTS_VEC3_H
#define DEV_QHC_CPP_PROJECTS_VEC3_H

#include <iostream>

using std::sqrt;

class Vec3 {
public:

    double e0{}, e1{}, e2{};

    __device__ Vec3() {}

    __device__ Vec3(double e0, double e1, double e2) {
        this->e0 = e0;
        this->e1 = e1;
        this->e2 = e2;
    }

    __device__  double x() const { return e0; }

    __device__ double y() const { return e1; }

    __device__ double z() const { return e2; }

    __device__ Vec3 operator-() const { return {-e0, -e1, -e2}; }

    __device__ double operator[](int i) const {
        switch (i) {
            case 0:
                return e0;
            case 1:
                return e1;
            case 2:
                return e2;

        }
        return e0;
    }

    __device__ double &operator[](int i) {
        switch (i) {
            case 0:
                return e0;
            case 1:
                return e1;
            case 2:
                return e2;

        }
        return e0;
    }

    __device__ Vec3 &operator+=(const Vec3 &v) {
        e0 += v.e0;
        e1 += v.e1;
        e2 += v.e2;
        return *this;
    }

    __device__ Vec3 &operator*=(const double t) {
        e0 *= t;
        e1 *= t;
        e2 *= t;
        return *this;
    }

    __device__ Vec3 &operator/=(const double t) {
        return *this *= 1 / t;
    }

    __device__ double length() const {
        return sqrt(length_squared());
    }

    __device__ double length_squared() const {
        return e0 * e0 + e1 * e1 + e2 * e2;
    }


    __device__ inline static Vec3 random() {
        return {random_double(), random_double(), random_double()};
    }

    __device__ inline static Vec3 random(double min, double max) {
        return {random_double(min, max), random_double(min, max), random_double(min, max)};
    }

    __device__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e0) < s) && (fabs(e1) < s) && (fabs(e2) < s);
    }
};

// Type aliases for Vec3
using Point3 = Vec3;   // 3D point
using Color = Vec3;    // RGB Color


//inline std::ostream &operator<<(std::ostream &out, const Vec3 &v) {
//    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
//}

__device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return {u.e0 + v.e0, u.e1 + v.e1, u.e2 + v.e2};
}

__device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
    return {u.e0 - v.e0, u.e1 - v.e1, u.e2 - v.e2};
}

__device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
    return {u.e0 * v.e0, u.e1 * v.e1, u.e2 * v.e2};
}

__device__ inline Vec3 operator*(double t, const Vec3 &v) {
    return {t * v.e0, t * v.e1, t * v.e2};
}

__device__ inline Vec3 operator*(const Vec3 &v, double t) {
    return t * v;
}

__device__ inline Vec3 operator/(Vec3 v, double t) {
    return (1 / t) * v;
}

__device__ inline double dot(const Vec3 &u, const Vec3 &v) {
    return u.e0 * v.e0
           + u.e1 * v.e1
           + u.e2 * v.e2;
}

__device__ inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
    return {u.e1 * v.e2 - u.e2 * v.e1,
            u.e2 * v.e0 - u.e0 * v.e2,
            u.e0 * v.e1 - u.e1 * v.e0};
}

__device__ inline Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}

__device__ Vec3 random_in_unit_sphere() {
    while (true) {
        auto p = Vec3::random(-1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}


__device__ Vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

//__device__ Vec3 random_in_hemisphere(const Vec3& normal) {
//    Vec3 in_unit_sphere = random_in_unit_sphere();
//    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
//        return in_unit_sphere;
//    else
//        return -in_unit_sphere;
//}

__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__device__ Vec3 refract(const Vec3 &uv, const Vec3 &n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ Vec3 random_in_unit_disk() {
    while (true) {
        auto p = Vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

#endif //DEV_QHC_CPP_PROJECTS_VEC3_H
