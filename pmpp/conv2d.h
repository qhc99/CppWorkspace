#ifndef PPMP_CONV2d_KERNEL_H
#define PPMP_CONV2d_KERNEL_H

/**
 * @brief convolution 2D, P = N * F, zero padding for N
 * 
 * @param N 
 * @param F 
 * @param P 
 * @param radius radius of filter F, default max 6, can be override in compile-time 
 * by -DPPMP_CONV2D_MAX_RADIUS 
 * @param width width of N and P
 * @param height height of N and P
 */
void conv2d(float *N, float *F, float *P, size_t radius, size_t width, size_t height);

#endif