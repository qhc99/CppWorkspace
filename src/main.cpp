
#include <iostream>

#include "utils.h"

using namespace org::nathan::utils;

int main() {

    auto t1 = currentTime();
    std::cout << "print\n";
    auto t2 = currentTime();
    std::cout << intervalToMilli(t2, t1) << "ms\n";
    return 0;
}
