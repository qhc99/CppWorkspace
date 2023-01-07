#include "rtweekend.h"
#include "write_color.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Camera.h"
#include "Material.h"
#include "utils.h"


__device__ Color ray_color(const Ray &r, const Hittable *world, int depth, curandState *state) {
    HitRecord rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0) {
        return {0, 0, 0};
    }
    if (world->hit(r, 0.001, infinity, rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, state)) {
            return attenuation * ray_color(scattered, world, depth - 1, state);
        }
        return {0, 0, 0};
    }

    Vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

constexpr int cache_size = 1000;

__device__ void random_scene(
    HittableList *world_dev,
    Sphere *sphere_cache,
    Hittable **hittable_cache,
    Lambertian *lambertian_cache,
    Metal *metal_cache,
    Dielectric *dielectric_cache,
    curandState *state
) {

    new(world_dev) HittableList(cache_size, hittable_cache);
    int sphere_idx = 0, lam_idx = 0, metal_idx = 0, dielec_idx = 0;

    new(&lambertian_cache[lam_idx])Lambertian(Color(0.5, 0.5, 0.5));
    auto ground_material = &lambertian_cache[lam_idx];
    lam_idx++;
    new(&sphere_cache[sphere_idx])Sphere(Point3(0, -1000, 0), 1000, ground_material);
    world_dev->add(&sphere_cache[sphere_idx]);
    sphere_idx++;

    // should consider cache size
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double(state);
            Point3 center(a + 0.9 * random_double(state), 0.2, b + 0.9 * random_double(state));

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                Material *sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = Color::random(state) * Color::random(state);

                    new(&lambertian_cache[lam_idx])Lambertian(albedo);
                    sphere_material = &lambertian_cache[lam_idx];
                    lam_idx++;
                    new(&sphere_cache[sphere_idx])Sphere(center, 0.2, sphere_material);
                    world_dev->add(&sphere_cache[sphere_idx]);
                    sphere_idx++;
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = Color::random(0.5, 1, state);
                    auto fuzz = random_double(0, 0.5, state);
                    new(&metal_cache[metal_idx])Metal(albedo, fuzz);
                    sphere_material = &metal_cache[metal_idx];
                    metal_idx++;
                    new(&sphere_cache[sphere_idx])Sphere(center, 0.2, sphere_material);
                    world_dev->add(&sphere_cache[sphere_idx]);
                    sphere_idx++;
                } else {
                    // glass
                    new(&dielectric_cache[dielec_idx]) Dielectric(1.5);
                    sphere_material = &dielectric_cache[dielec_idx];
                    dielec_idx++;
                    new(&sphere_cache[sphere_idx])Sphere(center, 0.2, sphere_material);
                    world_dev->add(&sphere_cache[sphere_idx]);
                    sphere_idx++;
                }
            }
        }
    }
    new(&dielectric_cache[dielec_idx])Dielectric(1.5);
    auto material1 = &dielectric_cache[dielec_idx];
    new(&sphere_cache[sphere_idx])Sphere(Point3(0, 1, 0), 1.0, material1);
    world_dev->add(&sphere_cache[sphere_idx]);
    sphere_idx++;

    new(&lambertian_cache[lam_idx])Lambertian(Color(0.4, 0.2, 0.1));
    auto material2 = &lambertian_cache[lam_idx];
    new(&sphere_cache[sphere_idx]) Sphere(Point3(-4, 1, 0), 1.0, material2);
    world_dev->add(&sphere_cache[sphere_idx]);
    sphere_idx++;

    new(&metal_cache[metal_idx])Metal(Color(0.7, 0.6, 0.5), 0.0);
    auto material3 = &metal_cache[metal_idx];
    new(&sphere_cache[sphere_idx])Sphere(Point3(4, 1, 0), 1.0, material3);
    world_dev->add(&sphere_cache[sphere_idx]);
}

constexpr auto aspect_ratio = 3.0 / 2.0;
constexpr int image_width = 1200; // 1200
constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
constexpr int samples_per_pixel = 500; // 500
constexpr int max_depth = 50;

__global__ void set_up(
    HittableList *world_dev,
    Sphere *sphere_cache,
    Hittable **hittable_cache,
    Lambertian *lambertian_cache,
    Metal *metal_cache,
    Dielectric *dielectric_cache,
    Camera *cam_dev,
    curandState *state
) {

    // init rand state
    curand_init(clock(), 0, 0, state);
    // World
    random_scene(
        world_dev,
        sphere_cache,
        hittable_cache,
        lambertian_cache,
        metal_cache,
        dielectric_cache,
        state);

    // Camera
    Point3 look_from(13, 2, 3);
    Point3 look_at(0, 0, 0);
    Vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    new(cam_dev) Camera(look_from, look_at, vup, 20, aspect_ratio, aperture, dist_to_focus);


}

__global__ void ray_trace(HittableList *world_dev, Camera *cam_dev, Color *color_store_dev, curandState *state) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    for (auto j = x; j < image_height; j += gridDim.x * blockDim.x) {
        for (auto i = y; i < image_width; i += gridDim.y * blockDim.y) {
            auto pixel_color = Color(0, 0, 0); // free when terminate
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double(state)) / (image_width - 1);
                auto v = (j + random_double(state)) / (image_height - 1);
                Ray r = cam_dev->get_ray(u, v, state);
                pixel_color += ray_color(r, world_dev, max_depth, state);
            }
            color_store_dev[j * image_width + i] = pixel_color;
        }
    }
}

int grid_dim(int desired, int block_dim, int up_limit) {
    if (desired * block_dim <= up_limit) {
        return desired;
    } else {
        return ceil(static_cast<float>(up_limit) / static_cast<float>(block_dim));
    }
}

int main() {

    const int block_dim_x_y = 16;
    dim3 block_dims(block_dim_x_y, block_dim_x_y);
    dim3 grid_dims(
        grid_dim(256, block_dim_x_y, image_height),
        grid_dim(256, block_dim_x_y, image_width));

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    // var
    auto *color_store = static_cast<Color *>(malloc(sizeof(Color) * image_width * image_height));
    Color *color_store_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&color_store_dev, sizeof(Color) * image_width * image_height));
    HittableList *world_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&world_dev, sizeof(HittableList)));
    Camera *cam_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&cam_dev, sizeof(Camera)));
    curandState *rnd_state_dev;
    HANDLE_ERROR(cudaMalloc(&rnd_state_dev, sizeof(curandState)));

    // cache
    Sphere *sphere_cache_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&sphere_cache_dev, sizeof(Sphere) * cache_size));
    Hittable **hittable_cache_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&hittable_cache_dev, sizeof(Hittable *) * cache_size));
    Lambertian *lambertian_cache_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&lambertian_cache_dev, sizeof(Lambertian) * cache_size));
    Metal *metal_cache_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&metal_cache_dev, sizeof(Metal) * cache_size));
    Dielectric *dielectric_cache_dev{nullptr};
    HANDLE_ERROR(cudaMalloc(&dielectric_cache_dev, sizeof(Dielectric) * cache_size));

    // set up
    set_up<<<1, 1>>>(
        world_dev,
        sphere_cache_dev,
        hittable_cache_dev,
        lambertian_cache_dev,
        metal_cache_dev,
        dielectric_cache_dev,
        cam_dev,
        rnd_state_dev);

    cudaDeviceSynchronize();
    // metric var
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));


    // kernel
    ray_trace<<<grid_dims, block_dims>>>(world_dev, cam_dev, color_store_dev, rnd_state_dev);

    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
                                      start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    // output image
    cudaMemcpy(color_store, color_store_dev, sizeof(Color) * image_height * image_width, cudaMemcpyDeviceToHost);
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            write_color(std::cout, color_store[j * image_width + i], samples_per_pixel);
        }
    }
    std::cerr << "\nDone.\n";

    HANDLE_ERROR(cudaFree(world_dev));
    HANDLE_ERROR(cudaFree(cam_dev));
    HANDLE_ERROR(cudaFree(color_store_dev));
    HANDLE_ERROR(cudaFree(rnd_state_dev));
    HANDLE_ERROR(cudaFree(sphere_cache_dev));
    HANDLE_ERROR(cudaFree(hittable_cache_dev));
    HANDLE_ERROR(cudaFree(lambertian_cache_dev));
    HANDLE_ERROR(cudaFree(metal_cache_dev));
    HANDLE_ERROR(cudaFree(dielectric_cache_dev));
    free(color_store);
}