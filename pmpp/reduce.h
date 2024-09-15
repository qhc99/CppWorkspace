#ifndef PPMP_REDUCE_KERNEL_H
#define PPMP_REDUCE_KERNEL_H


void reduce_min_i(const int* input, size_t length, int* output);

void reduce_max_i(const int* input, size_t length, int* output);

void reduce_add_f(const float* input, size_t length, float* output);
#endif