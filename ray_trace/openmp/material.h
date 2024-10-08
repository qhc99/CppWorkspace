#ifndef DEV_QC_CPP_PROJECTS_MATERIAL_H
#define DEV_QC_CPP_PROJECTS_MATERIAL_H

#include "hittable.h"
#include "rtweekend.h"

class Material {
public:
    Material() = default;
    Material(const Material& o) = default;
    Material(Material&& o) = default;
    Material& operator=(const Material& o) = default;
    Material& operator=(Material&& o) = default;
    virtual ~Material() = default;
    virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const
        = 0;
};

class Lambertian : public Material {
public:
    explicit Lambertian(const Color& a)
        : albedo(a)
    {
    }

    bool scatter(
        const Ray& /*r_in*/, const HitRecord& rec, Color& attenuation, Ray& scattered) const override
    {
        auto scatter_direction = rec.normal + random_unit_vector();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }

        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

    Color albedo;
};

class Metal : public Material {
public:
    Metal(const Color& a, double f)
        : albedo(a)
        , fuzz(f < 1 ? f : 1)
    {
    }

    bool scatter(
        const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override
    {
        Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    Color albedo;
    double fuzz;
};

class Dielectric : public Material {
public:
    explicit Dielectric(double index_of_refraction)
        : ir(index_of_refraction)
    {
    }

    bool scatter(
        const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override
    {
        attenuation = Color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        Vec3 unit_direction = unit_vector(r_in.direction());
        Vec3 refracted = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = Ray(rec.p, refracted);
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = Ray(rec.p, direction);
        return true;
    }

    double ir; // Index of Refraction
private:
    static double reflectance(double cosine, double ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif // DEV_QC_CPP_PROJECTS_MATERIAL_H
