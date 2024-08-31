#ifndef PMPP_MAT_MUL_KERNEL_H
#define PMPP_MAT_MUL_KERNEL_H

/**
 * @brief Mat Mul Tiling and Coarsening. Available tune compile definitions: 
 * PMPP_MAT_MUL_KERNEL_TILE_WIDTH, PMPP_MAT_MUL_KERNEL_COARSE_FACTOR
 *
 * @param A row major matrix, size i * j
 * @param B row major matrix, size j * k
 * @param C row major matrix, size i * k, return C = A * B
 * @param i
 * @param j
 * @param k
 * @return void
 */
void matMul(float* A, float* B, float* C, size_t i, size_t j, size_t k);

#endif