#ifndef PPMP_CONV2d_KERNEL_H
#define PPMP_CONV2d_KERNEL_H

/**
 * @brief convolution 2D, P = N * F, zero padding for N,
 * 
 * @param N input
 * @param F filter
 * @param P output
 * @param radius radius of filter F, limited by compile definition PPMP_CONV2D_FILTER_MAX_RADIUS
 * by -DPPMP_CONV2D_MAX_RADIUS 
 * @param width width of N and P
 * @param height height of N and P
 */
void conv2d(float *N, float *F, float *P, size_t radius, size_t width, size_t height);

#endif