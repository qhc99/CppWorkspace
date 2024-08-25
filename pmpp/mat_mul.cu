#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "utils.h"

#define MAT_MUL_KERNEL_TILE_WIDTH 16

__global__ void matMulKernel(float* A, float* B, float* C, size_t i, size_t j, size_t k)
{

}

void matMul(float* A, float* B, float* C, size_t i, size_t j, size_t k)
{
    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;

    checkCudaErrors(cudaMalloc(&A_d, i * j * sizeof(float)));
    checkCudaErrors(cudaMalloc(&B_d, j * k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&C_d, i * k * sizeof(float)));

    checkCudaErrors(cudaMemcpy(A_d, A, i * j * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, j * k * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_dim { MAT_MUL_KERNEL_TILE_WIDTH, MAT_MUL_KERNEL_TILE_WIDTH ,1};

    matMulKernel<<<128, block_dim>>>(A_d, B_d, C_d, i, j, k);

    checkCudaErrors(cudaMemcpy(C, C_d, i * k * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}