#include "utils.h"

constexpr size_t TILE_WIDTH { 16 };

// Tiling
__global__ void matMulKernel(float* A, float* B, float* C, size_t i, size_t j, size_t k)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    size_t bx { blockIdx.x };
    size_t by { blockIdx.y };
    size_t tx { threadIdx.x };
    size_t ty { threadIdx.y };

    size_t row = bx * TILE_WIDTH + tx;
    size_t col = by * TILE_WIDTH + ty;

    float p {};

    for (size_t ph {}; static_cast<float>(ph) < ceil(static_cast<float>(j) / static_cast<float>(TILE_WIDTH)); ph++) {
        Mds[tx][ty] = (row < i && (ph * TILE_WIDTH + ty) < j) ? A[row * j + ph * TILE_WIDTH + ty] : 0;
        Nds[tx][ty] = ((ph * TILE_WIDTH + tx) < j && col < k) ? B[(ph * TILE_WIDTH + tx) * k + col] : 0;
        __syncthreads();

        for (size_t k {}; k < TILE_WIDTH; k++) {
            p += Mds[tx][k] * Nds[k][ty];
        }
        __syncthreads();
    }
    if (row < i && col < k) {
        C[row * k + col] = p;
    }
}

void matMul(float* A, float* B, float* C, size_t i, size_t j, size_t k)
{
    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;

    checkCudaErrors(cudaMalloc(&A_d, i * j * sizeof(float)));
    checkCudaErrors(cudaMalloc(&B_d, j * k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&C_d, i * k * sizeof(float)));

    checkCudaErrors(cudaMemcpy(A_d, A, i * j * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_d, B, j * k * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_dim { TILE_WIDTH, TILE_WIDTH, 1 };
    auto grid_dim_x { static_cast<unsigned int>(std::ceil(static_cast<float>(i) / static_cast<float>(TILE_WIDTH))) };
    auto grid_dim_y { static_cast<unsigned int>(std::ceil(static_cast<float>(k) / static_cast<float>(TILE_WIDTH))) };
    dim3 grid_dim { grid_dim_x, grid_dim_y, 1 };
    matMulKernel<<<grid_dim, block_dim>>>(A_d, B_d, C_d, i, j, k);

    checkCudaErrors(cudaMemcpy(C, C_d, i * k * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(A_d));
    checkCudaErrors(cudaFree(B_d));
    checkCudaErrors(cudaFree(C_d));
}