//
// Created by QC on 2022-12-10.
//
#include "lib_central/utils.h"
#include <cmath>
#include <random>
#include <iostream>
#include <chrono>
#include <execution>

using dev::qhc::utils::default_rand_engine;
using std::chrono::nanoseconds;
using dev::qhc::utils::current_time_point;
using dev::qhc::utils::time_point_duration_to_us;


struct MatBlock {
    const int r1;
    const int c1;
    const int r2;
    const int c2;
};

class MatPartition {

private:
    const int p;

    const int m;
    const int n;

public:
    const int rows;
    const int cols;

    MatPartition(int parti, int m, int n)
        : p{parti},
          m{m},
          n{n},
          rows{static_cast<int>(std::ceil(static_cast<float>(m) / static_cast<float> (parti)))},
          cols{static_cast<int>(std::ceil(static_cast<float>(n) / static_cast<float> (parti)))} {

    }


    [[nodiscard]] MatBlock at(int i, int j) const {
        return {i * p, j * p,
                std::min(i * p + p, m), std::min(j * p + p, n)};
    }
};

static void mul_mat_block(float **m1, MatBlock m1r,
                          float **m2, MatBlock m2r,
                          float **m3, MatBlock m3r) {
    for (int i = 0; i + m3r.r1 < m3r.r2; i++) {
        for (int j = 0; j + m3r.c1 < m3r.c2; j++) {
            for (int k = 0; k + m2r.r1 < m2r.r2; k++) {
                m3[i + m3r.r1][j + m3r.c1] +=
                    m1[i + m1r.r1][k + m1r.c1] * m2[k + m2r.r1][j + m2r.c1];
            }
        }
    }
}

void reset(float **m1, float **m2, float **m3, int size_a, int size_b, int size_c) {
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
}

void space_locality(float **m1, float **m2, float **m3, const int size_a, const int size_b, const int size_c,
                    int w_s) {
    auto &&t3{current_time_point()};
    const MatPartition a1(w_s, size_a, size_b);
    const MatPartition a2(w_s, size_b, size_c);
    const MatPartition a3(w_s, size_a, size_c);

    std::vector<int> loop1(a3.rows);
    for (int i = 0; i < a3.rows; i++) {
        loop1.at(i) = i;
    }

    std::for_each(
        std::execution::par_unseq,
        loop1.begin(),
        loop1.end(),
        [&](auto i) {
            for (int j = 0; j < a3.cols; j++) {
                MatBlock m3r = a3.at(i, j);
                for (int p = 0; p + m3r.r1 < m3r.r2; p++) {
                    for (int q = 0; q + m3r.c1 < m3r.c2; q++) {
                        m3[p + m3r.r1][q + m3r.c1] = 0;
                    }
                }
                for (int k = 0; k < a1.cols; k++) {
                    MatBlock m1r = a1.at(i, k);
                    MatBlock m2r = a2.at(k, j);
                    mul_mat_block(m1, m1r, m2, m2r, m3, m3r);
                }
            }
        });
    auto &&t4{current_time_point()};

    double ans = 0;
    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_c; j++) {
            ans += m3[i][j];
        }
    }

    std::cout << "\n";
    std::cout << "locality" << std::endl;
    std::cout << "spend: " << time_point_duration_to_us(t4, t3) / 1000000. << " s" << std::endl;
    std::cout << "sum: " << ans << std::endl;
}

void space_locality_openmp(float **m1, float **m2, float **m3, const int size_a, const int size_b, const int size_c,
                           int w_s) {
    MatPartition a1(w_s, size_a, size_b);
    MatPartition a2(w_s, size_b, size_c);
    MatPartition a3(w_s, size_a, size_c);
    auto &&t5{current_time_point()};

#pragma omp parallel for default(none) shared(a1, a2, a3, m1, m2, m3)
    for (int i = 0; i < a3.rows; i++) {
        for (int j = 0; j < a3.cols; j++) {
            MatBlock m3r = a3.at(i, j);
            for (int p = 0; p + m3r.r1 < m3r.r2; p++) {
                for (int q = 0; q + m3r.c1 < m3r.c2; q++) {
                    m3[p + m3r.r1][q + m3r.c1] = 0;
                }
            }
            for (int k = 0; k < a1.cols; k++) {
                MatBlock m1r = a1.at(i, k);
                MatBlock m2r = a2.at(k, j);
                mul_mat_block(m1, m1r, m2, m2r, m3, m3r);
            }
        }
    }


    auto &&t6{current_time_point()};

    double ans = 0;
    for (int i = 0; i < size_a; i++) {
        for (int j = 0; j < size_c; j++) {
            ans += m3[i][j];
        }
    }

    std::cout << "\n";
    std::cout << "locality openmp" << std::endl;
    std::cout << "spend: " << time_point_duration_to_us(t6, t5) / 1000000. << " s" << std::endl;
    std::cout << "sum: " << ans << std::endl;
}

void run_demo(){
    constexpr int size_a = 1500;
    constexpr int size_b = 1500;
    constexpr int size_c = 1500;
    auto **m1 = new float *[size_a];
    auto **m2 = new float *[size_b];
    auto **m3 = new float *[size_a];


    reset(m1, m2, m3, size_a, size_b, size_c);
    space_locality(m1, m2, m3, size_a, size_b, size_c, 64);

    reset(m1, m2, m3, size_a, size_b, size_c);
    space_locality_openmp(m1, m2, m3, size_a, size_b, size_c, 64);
}

int main(){
    run_demo();
    return 0;
}