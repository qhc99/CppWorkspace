#include "rtweekend.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include <execution>
#include "lib_central/utils.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
Color ray_color(const Ray &r, const Hittable &world, int depth) {
    HitRecord rec;

    // If we've exceeded the Ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return {0, 0, 0};

    if (world.hit(r, 0.001, infinity, rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return {0, 0, 0};
    }

    Vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}
#pragma clang diagnostic pop

HittableList random_scene() {
    HittableList world;

    auto ground_material = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            Point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<Material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = Color::random() * Color::random();
                    sphere_material = make_shared<Lambertian>(albedo);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // Metal
                    auto albedo = Color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<Dielectric>(1.5);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<Dielectric>(1.5);
    world.add(make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));

    return world;
}

int main() {

    // Image

    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200; // 1200
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 500; // 500
    const int max_depth = 50;

    // World

    auto world = random_scene();

    // Camera

    Point3 lookfrom(13, 2, 3);
    Point3 lookat(0, 0, 0);
    Vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    // Render

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    //---------------------------------------------------
    auto **color_store = new Color *[image_height];
    for (int i = 0; i < image_height; i++) {
        color_store[i] = new Color[image_width];
    }
    auto t1{dev::qhc::utils::current_time_point()};
#pragma omp parallel for default(none) shared(image_height, world, cam, color_store,std::cerr)
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto *pixel_color = new Color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                Ray r = cam.get_ray(u, v);
                *pixel_color += ray_color(r, world, max_depth);
            }
            color_store[j][i] = *pixel_color;
        }
        std::cerr << j << std::endl;
    }

    auto t2{dev::qhc::utils::current_time_point()};

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            write_color(std::cout, color_store[j][i], samples_per_pixel);
        }
    }
    std::cerr << "Spend " << dev::qhc::utils::time_point_duration_to_us(t2,t1)/1000000.0 << "s" << std::endl;
    std::cerr << "\nDone.\n";

    for (int i = 0; i < image_height; i++) {
        delete[] color_store[i];
    }
    delete[] color_store;
}