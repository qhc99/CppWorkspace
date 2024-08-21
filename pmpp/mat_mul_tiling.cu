#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "utils.h"

#define MAT_MUL_KERNEL_TILE_WIDTH 16

__global__ void matMulTilingKernel(float* A, float* B, float* C, int i, int j, int k)
{

}

void matMulTiling(float* A, float* B, float* C, int i, int j, int k)
{
    float *A_d, *B_d, *C_d;

    checkCudaErrors(cudaMalloc((void**)&A_d, i * j * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&B_d, j * k * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&C_d, i * k * sizeof(float)));

    checkCudaErrors(cudaMemcpy(A_d, A, i * j * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, j * k * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_dim { MAT_MUL_KERNEL_TILE_WIDTH, MAT_MUL_KERNEL_TILE_WIDTH ,1};

    matMulTilingKernel<<<128, block_dim>>>(A_d, B_d, C_d, i, j, k);

    checkCudaErrors(cudaMemcpy(C, C_d, i * k * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}