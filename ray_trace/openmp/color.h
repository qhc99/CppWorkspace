//
// Created by QC on 2022-12-05.
//

#ifndef DEV_QHC_CPP_PROJECTS_WRITE_COLOR_H
#define DEV_QHC_CPP_PROJECTS_COLOR_H

#include "vec3.h"

#include "../../../../usr/include/c++/11/iostream"

void write_color(std::ostream &out, Color pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the Color by the number of samples and gamma-correct for gamma=2.0.
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0,255] value of each Color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

#endif //DEV_QHC_CPP_PROJECTS_WRITE_COLOR_H
