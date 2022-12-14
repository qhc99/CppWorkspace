//
// Created by Nathan on 2022-12-10.
//
#include <random>
#include <iostream>
#include "lib_central/utils.h"
#include <chrono>
#include <cassert>
#include <execution>
#include <cmath>

using dev::qhc::utils::default_rand_engine;
using std::chrono::duration_cast, std::chrono::nanoseconds;
using dev::qhc::utils::current_time_point;
using dev::qhc::utils::time_point_duration_to_ms;


struct MatRange {
    int r1;
    int c1;
    int r2;
    int c2;
};

class PartitionMat {

private:
    const int p;

    const int m;
    const int n;

public:
    const int rows;
    const int cols;

    PartitionMat(float **mat, int parti, int m, int n)
        : p{parti},
          m{m},
          n{n},
          rows{static_cast<int>(ceil(m / (float) parti))},
          cols{static_cast<int>(ceil(n / (float) parti))} {

    }


    MatRange at(int i, int j) {
        return {i * p, j * p,
                std::min(i * p + p, m), std::min(j * p + p, n)};
    }
};

static void mul(float **m1, MatRange m1r,
                float **m2, MatRange m2r,
                float **m3, MatRange m3r) {
    for (int i = 0; i + m3r.r1 < m3r.r2; i++) {
        for (int j = 0; j + m3r.c1 < m3r.c2; j++) {
            for (int k = 0; k + m2r.r1 < m2r.r2; k++) {
                m3[i + m3r.r1][j + m3r.c1] +=
                    m1[i + m1r.r1][k + m1r.c1] * m2[k + m2r.r1][j + m2r.c1];
            }
        }
    }
}

int main() {
    constexpr int size_a = 1700;
    constexpr int size_b = 1700;
    constexpr int size_c = 1700;
    float **m1 = new float *[size_a];
    float **m2 = new float *[size_b];
    float **m3 = new float *[size_a];
    std::uniform_real_distribution<float> dist(0, 1);
    auto rand_engine{default_rand_engine()};
    for (int i = 0; i < size_a; i++) {
        m1[i] = new float[size_b];
        m3[i] = new float[size_c];
    }
    for (int i = 0; i < size_b; i++) {
        m2[i] = new float[size_c];
    }


    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_b; j++) {
            m1[i][j] = dist(rand_engine);
        }
    }
    for (int i = 0; i < size_b; i++) {
        for (int j = 0; j < size_c; j++) {
            m2[i][j] = dist(rand_engine);
        }
    }
    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_c; j++) {
            m3[i][j] = 0;
        }
    }
    std::cout << "start" << std::endl;
    auto &&t1{current_time_point()};
    std::vector<int> loop(size_a);
    for (int i = 0; i < size_a; i++) {
        loop.at(i) = i;
    }

    std::for_each(
        std::execution::par,
        loop.begin(),
        loop.end(),
        [&](auto &&i) {
            for (int j = 0; j < size_c; j++) {
                for (int k = 0; k < size_b; k++) {
                    m3[i][j] += m1[i][k] * m2[k][j];
                }
            }
        });
    auto &&t2{current_time_point()};
    double ans = 0;
    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_c; j++) {
            ans += m3[i][j];
        }
    }

    std::cout << "naive par" << std::endl;
    std::cout << "spend: " << time_point_duration_to_ms(t2, t1) / 1000. << " ms" << std::endl;
    std::cout << "sum: " << ans << std::endl;


    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_c; j++) {
            m3[i][j] = 0;
        }
    }
    auto &&t3{current_time_point()};
    int w_s = 32;
    PartitionMat a1(m1, w_s, size_a, size_b);
    PartitionMat a2(m2, w_s, size_b, size_c);
    PartitionMat a3(m3, w_s, size_a, size_c);

    std::vector<int> loop1(a3.rows);
    for (int i = 0; i < a3.rows; i++) {
        loop.at(i) = i;
    }

    std::for_each(
        std::execution::par,
        loop.begin(),
        loop.end(),
        [&](auto &&i) {
            for (int j = 0; j < a3.cols; j++) {
                MatRange m3r = a3.at(i, j);
                for (int k = 0; k < a1.cols; k++) {
                    MatRange m1r = a1.at(i, k);
                    MatRange m2r = a2.at(k, j);
                    mul(m1, m1r, m2, m2r, m3, m3r);
                }
            }
        });
    auto &&t4{current_time_point()};

    ans = 0;
    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_c; j++) {
            ans += m3[i][j];
        }
    }

    std::cout << "locality par" << std::endl;
    std::cout << "spend: " << time_point_duration_to_ms(t4, t3) / 1000. << " ms" << std::endl;
    std::cout << "sum: " << ans << std::endl;
}