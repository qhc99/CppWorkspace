//
// Created by Nathan on 2022-12-10.
//
#include <random>
#include <iostream>
#include "lib_central/utils.h"
#include <chrono>
#include <cassert>

using dev::qhc::utils::default_rand_engine;
using std::chrono::duration_cast, std::chrono::nanoseconds;
using dev::qhc::utils::current_time_point;
using dev::qhc::utils::time_point_duration_to_ms;

int main(){
    assert(CHAR_BIT * sizeof (float) == 32);
    constexpr int size = 1300;
    float **m1 = new float*[size];
    float **m2 = new float*[size];
    float **m3 = new float*[size];
    std::uniform_real_distribution<float> dist(0, 1);
    auto rand_engine{default_rand_engine()};
    for (int i = 0; i < size; i++) {
        m1[i] = new float [size];
        m2[i] = new float [size];
        m3[i] = new float [size];
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            m1[i][j] = dist(rand_engine);
            m2[i][j] = dist(rand_engine);
            m3[i][j] = 0;
        }
    }
    auto t1{current_time_point()};
    for (int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++){
            for (int k = 0; k < size; k++) {
                m3[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
    auto t2{current_time_point()};
    double ans = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            ans += m3[i][j];
        }
    }

    std::cout << time_point_duration_to_ms(t2, t1) / 1000. << std::endl;
    std::cout << ans << std::endl;
}