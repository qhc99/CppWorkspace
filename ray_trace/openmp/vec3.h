#ifndef DEV_QC_CPP_PROJECTS_VEC3_H
#define DEV_QC_CPP_PROJECTS_VEC3_H

#include "rtweekend.h"


using std::sqrt;

class Vec3 {
public:
    Vec3()
        : e { 0, 0, 0 }
    {
    }

    Vec3(double e0, double e1, double e2)
        : e { e0, e1, e2 }
    {
    }

    [[nodiscard]] double x() const { return e[0]; }

    [[nodiscard]] double y() const { return e[1]; }

    [[nodiscard]] double z() const { return e[2]; }

    Vec3 operator-() const { return {-e[0], -e[1], -e[2]}; }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    double operator[](int i) const { return e[i]; }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    double& operator[](int i) { return e[i]; }

    Vec3& operator+=(const Vec3& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    Vec3& operator*=(const double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    Vec3& operator/=(const double t)
    {
        return *this *= 1 / t;
    }

    [[nodiscard]] double length() const
    {
        return sqrt(length_squared());
    }

    [[nodiscard]] double length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    std::array<double, 3> e;

    inline static Vec3 random()
    {
        return { random_double(), random_double(), random_double() };
    }

    inline static Vec3 random(double min, double max)
    {
        return { random_double(min, max), random_double(min, max), random_double(min, max) };
    }

    [[nodiscard]] bool near_zero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }
};

// Type aliases for Vec3
using Point3 = Vec3; // 3D point
using Color = Vec3; // RGB Color

inline std::ostream& operator<<(std::ostream& out, const Vec3& v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline Vec3 operator+(const Vec3& u, const Vec3& v)
{
    return { u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2] };
}

inline Vec3 operator-(const Vec3& u, const Vec3& v)
{
    return { u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2] };
}

inline Vec3 operator*(const Vec3& u, const Vec3& v)
{
    return { u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2] };
}

inline Vec3 operator*(double t, const Vec3& v)
{
    return { t * v.e[0], t * v.e[1], t * v.e[2] };
}

inline Vec3 operator*(const Vec3& v, double t)
{
    return t * v;
}

inline Vec3 operator/(Vec3 v, double t)
{
    return (1 / t) * v;
}

inline double dot(const Vec3& u, const Vec3& v)
{
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

inline Vec3 cross(const Vec3& u, const Vec3& v)
{
    return { u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0] };
}

inline Vec3 unit_vector(Vec3 v)
{
    return v / v.length();
}

inline Vec3 random_in_unit_sphere()
{
    while (true) {
        auto p = Vec3::random(-1, 1);
        if (p.length_squared() >= 1) {
            continue;
        }
        return p;
    }
}

inline Vec3 random_unit_vector()
{
    return unit_vector(random_in_unit_sphere());
}

inline Vec3 random_in_hemisphere(const Vec3& normal)
{
    Vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0) { // In the same hemisphere as the normal
        return in_unit_sphere;
    } else {
        return -in_unit_sphere;
    }
}

inline Vec3 reflect(const Vec3& v, const Vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

inline Vec3 refract(const Vec3& uv, const Vec3& n, double etai_over_etat)
{
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

inline Vec3 random_in_unit_disk()
{
    while (true) {
        auto p = Vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() >= 1) {
            continue;
        }
        return p;
    }
}

#endif // DEV_QC_CPP_PROJECTS_VEC3_H
