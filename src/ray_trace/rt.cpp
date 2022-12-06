
#include "lib_central/utils.h"
#include <iostream>


using dev::qhc::utils::current_time_point;
using dev::qhc::utils::time_point_interval_to_ms;
using dev::qhc::utils::default_rand_engine;


auto mt{default_rand_engine()};
std::uniform_real_distribution<float> dist(0.0, 1.0);

void print_pixels(){
    // Image

    constexpr int image_width = 256;
    constexpr int image_height = 256;

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    std::cerr << "\nDone.\n";
}

int main() {
    print_pixels();
}
