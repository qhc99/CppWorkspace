#include <cstddef>

#include "utils.h"

#ifndef PMPP_MAT_MUL_KERNEL_TILE_WIDTH
#define PMPP_MAT_MUL_KERNEL_TILE_WIDTH 16
#endif

#ifndef PMPP_MAT_MUL_KERNEL_COARSE_FACTOR
#define PMPP_MAT_MUL_KERNEL_COARSE_FACTOR 2
#endif

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
__global__ void matMulKernel(const float* A, const float* B, float* C, size_t i, size_t j, size_t k)
{
    extern __shared__ float shared_mem[];
    float* Mds { shared_mem };
    float* Nds { shared_mem + static_cast<unsigned int>(PMPP_MAT_MUL_KERNEL_TILE_WIDTH * PMPP_MAT_MUL_KERNEL_TILE_WIDTH) };

    size_t bx { blockIdx.x };
    size_t by { blockIdx.y };
    size_t tx { threadIdx.x };
    size_t ty { threadIdx.y };

    size_t row = bx * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + tx;
    size_t colStart = by * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + ty;

    float ps[PMPP_MAT_MUL_KERNEL_COARSE_FACTOR];
    for (size_t t {}; t < PMPP_MAT_MUL_KERNEL_COARSE_FACTOR; ++t) {
        ps[t] = 0;
    }

    for (size_t ph {}; static_cast<float>(ph) < ceil(static_cast<float>(j) / static_cast<float>(PMPP_MAT_MUL_KERNEL_TILE_WIDTH)); ph++) {
        Mds[tx * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + ty] = (row < i && (ph * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + ty) < j) ? A[row * j + ph * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + ty] : 0;

        for (size_t c {}; c < PMPP_MAT_MUL_KERNEL_COARSE_FACTOR; ++c) {
            size_t col { colStart + c * PMPP_MAT_MUL_KERNEL_TILE_WIDTH };
            Nds[tx * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + ty] = ((ph * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + tx) < j && col < k) ? B[(ph * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + tx) * k + col] : 0;
            __syncthreads();

            for (size_t k {}; k < PMPP_MAT_MUL_KERNEL_TILE_WIDTH; k++) {
                ps[c] += Mds[tx * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + k] * Nds[k * PMPP_MAT_MUL_KERNEL_TILE_WIDTH + ty];
            }
            __syncthreads();
        }
    }
    for (size_t c {}; c < PMPP_MAT_MUL_KERNEL_COARSE_FACTOR; ++c) {
        size_t col { colStart + c * PMPP_MAT_MUL_KERNEL_TILE_WIDTH };
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

    dim3 block_dim { PMPP_MAT_MUL_KERNEL_TILE_WIDTH, PMPP_MAT_MUL_KERNEL_TILE_WIDTH, 1 };
    auto grid_dim_x { static_cast<unsigned int>(std::ceil(static_cast<float>(i) / static_cast<float>(PMPP_MAT_MUL_KERNEL_TILE_WIDTH))) };
    auto grid_dim_y { static_cast<unsigned int>(std::ceil(static_cast<float>(k) / static_cast<float>(PMPP_MAT_MUL_KERNEL_TILE_WIDTH))) };
    dim3 grid_dim { grid_dim_x, grid_dim_y, 1 };
    matMulKernel<<<grid_dim, block_dim, static_cast<unsigned long>(2 * PMPP_MAT_MUL_KERNEL_TILE_WIDTH * PMPP_MAT_MUL_KERNEL_TILE_WIDTH) * sizeof(float)>>>(A_d, B_d, C_d, i, j, k);
    checkCudaLastError();
    checkCudaError(cudaMemcpy(C, C_d, i * k * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(A_d));
    checkCudaError(cudaFree(B_d));
    checkCudaError(cudaFree(C_d));
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)