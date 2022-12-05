
#include "lib_central/utils.h"
#include <iostream>



using dev::qhc::utils::currentTime;
using dev::qhc::utils::timeIntervalToMilli;
using dev::qhc::utils::random_engine;


auto mt{random_engine()};
std::uniform_real_distribution<double> dist(0.0, 1.0);
float rand_d() {
    return static_cast<float>(dist(mt));
}

int main()
{

    const int size = 1300;
    auto** m1 = new float* [size];
    auto** m2 = new float* [size];
    auto** m3 = new float* [size];

    for (int i = 0; i < size; i++) {
        m1[i] = new float[size];
        m2[i] = new float[size];
        m3[i] = new float[size];
    }



    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            m1[i][j] = rand_d();
            m2[i][j] = rand_d();
            m3[i][j] = 0;
        }
    }

    auto t1{ currentTime() };
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                m3[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    auto t2{ currentTime() };
    std::cout << timeIntervalToMilli(t2, t1) << std::endl;


}
