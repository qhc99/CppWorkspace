#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "utils.h"

__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n)
{
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&A_d, size));
    checkCudaErrors(cudaMalloc((void**)&B_d, size));
    checkCudaErrors(cudaMalloc((void**)&C_d, size));

    checkCudaErrors(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    // Cannot handle large n
    vecAddKernel<<<static_cast<unsigned int>(ceil(n / 256.)), 256>>>(A_d, B_d, C_d, n);

    checkCudaErrors(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}