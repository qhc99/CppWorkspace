#include "utils.h"

constexpr size_t TILE_WIDTH { 32 };
constexpr size_t COARSE_FACTOR { 2 };

__global__ void matMulKernel(float* A, float* B, float* C, size_t i, size_t j, size_t k)
{
    extern __shared__ float shared_mem[];
    float* Mds { shared_mem };
    float* Nds { shared_mem + TILE_WIDTH * TILE_WIDTH };

    size_t bx { blockIdx.x };
    size_t by { blockIdx.y };
    size_t tx { threadIdx.x };
    size_t ty { threadIdx.y };

    size_t row = bx * TILE_WIDTH + tx;
    size_t colStart = by * TILE_WIDTH + ty;

    float ps[COARSE_FACTOR];
    for (size_t t {}; t < COARSE_FACTOR; ++t) {
        ps[t] = 0;
    }

    for (size_t ph {}; static_cast<float>(ph) < ceil(static_cast<float>(j) / static_cast<float>(TILE_WIDTH)); ph++) {
        Mds[tx * TILE_WIDTH + ty] = (row < i && (ph * TILE_WIDTH + ty) < j) ? A[row * j + ph * TILE_WIDTH + ty] : 0;

        for (size_t c {}; c < COARSE_FACTOR; ++c) {
            size_t col { colStart + c * TILE_WIDTH };
            Nds[tx * TILE_WIDTH + ty] = ((ph * TILE_WIDTH + tx) < j && col < k) ? B[(ph * TILE_WIDTH + tx) * k + col] : 0;
            __syncthreads();

            for (size_t k {}; k < TILE_WIDTH; k++) {
                ps[c] += Mds[tx * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + ty];
            }
            __syncthreads();
        }
    }
    for (size_t c {}; c < COARSE_FACTOR; ++c) {
        size_t col { colStart + c * TILE_WIDTH };
        if (row < i && col < k) {
            C[row * k + col] = ps[c];
        }
    }
}

void matMul(float* A, float* B, float* C, size_t i, size_t j, size_t k)
{
    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;

    checkCudaError(cudaMalloc(&A_d, i * j * sizeof(float)));
    checkCudaError(cudaMalloc(&B_d, j * k * sizeof(float)));
    checkCudaError(cudaMalloc(&C_d, i * k * sizeof(float)));

    checkCudaError(cudaMemcpy(A_d, A, i * j * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(B_d, B, j * k * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_dim { TILE_WIDTH, TILE_WIDTH, 1 };
    auto grid_dim_x { static_cast<unsigned int>(std::ceil(static_cast<float>(i) / static_cast<float>(TILE_WIDTH))) };
    auto grid_dim_y { static_cast<unsigned int>(std::ceil(static_cast<float>(k) / static_cast<float>(TILE_WIDTH))) };
    dim3 grid_dim { grid_dim_x, grid_dim_y, 1 };
    matMulKernel<<<grid_dim, block_dim, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(A_d, B_d, C_d, i, j, k);
    checkCudaLastError();
    checkCudaError(cudaMemcpy(C, C_d, i * k * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(A_d));
    checkCudaError(cudaFree(B_d));
    checkCudaError(cudaFree(C_d));
}