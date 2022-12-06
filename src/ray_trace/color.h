//
// Created by Nathan on 2022-12-05.
//

#ifndef DEV_QHC_CPP_PROJECTS_COLOR_H
#define DEV_QHC_CPP_PROJECTS_COLOR_H

#include "vec3.h"

#include <iostream>

void write_color(std::ostream &out, color pixel_color) {
    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(255.999 * pixel_color.x()) << ' '
        << static_cast<int>(255.999 * pixel_color.y()) << ' '
        << static_cast<int>(255.999 * pixel_color.z()) << '\n';
}

#endif //DEV_QHC_CPP_PROJECTS_COLOR_H
