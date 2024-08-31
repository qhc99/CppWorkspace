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

    checkCudaError(cudaMalloc(&A_d, size));
    checkCudaError(cudaMalloc(&B_d, size));
    checkCudaError(cudaMalloc(&C_d, size));

    checkCudaError(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    vecAddKernel<<<static_cast<unsigned int>(ceil(static_cast<double>(n) / 256.)), 256>>>(A_d, B_d, C_d, n);
    checkCudaLastError();
    checkCudaError(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    checkCudaError(cudaFree(A_d));
    checkCudaError(cudaFree(B_d));
    checkCudaError(cudaFree(C_d));
}