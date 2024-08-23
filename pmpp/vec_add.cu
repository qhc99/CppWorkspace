#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "utils.h"

__global__ void vecAddKernel(float* A, float* B, float* C, size_t n)
{
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, size_t n)
{
    float *A_d = nullptr;
    float *B_d = nullptr;
    float *C_d = nullptr;
    size_t size = n * sizeof(float);

    checkCudaErrors(cudaMalloc(&A_d, size));
    checkCudaErrors(cudaMalloc(&B_d, size));
    checkCudaErrors(cudaMalloc(&C_d, size));

    checkCudaErrors(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    // Cannot handle large n
    vecAddKernel<<<static_cast<unsigned int>(ceil(static_cast<double>(n) / 256.)), 256>>>(A_d, B_d, C_d, n);

    checkCudaErrors(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}