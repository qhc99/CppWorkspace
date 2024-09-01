#define PPMP_CONV2D_FILTER_MAX_RADIUS 5
#define PPMP_CONV2D_TILE_WIDTH 32

#include "utils.h"

__constant__ float filter[(PPMP_CONV2D_FILTER_MAX_RADIUS * 2 + 1) * (PPMP_CONV2D_FILTER_MAX_RADIUS * 2 + 1)];

__global__ void conv2dKernel(float* N, float* P, size_t radius, size_t width, size_t height)
{
    size_t row = blockIdx.x * PPMP_CONV2D_TILE_WIDTH + threadIdx.x;
    size_t col = blockIdx.y * PPMP_CONV2D_TILE_WIDTH + threadIdx.y;
    constexpr size_t tile_width = PPMP_CONV2D_TILE_WIDTH;
    __shared__ float N_s[tile_width][tile_width];
    size_t threadX { threadIdx.x };
    size_t threadY { threadIdx.y };
    if (row < height && col < width) {
        N_s[threadX][threadY] = N[row * width + col];
    } else {
        N_s[threadX][threadY] = 0;
    }

    __syncthreads();

    if (row >= height || col >= width) {
        return;
    }

    float val {};
    for (size_t fr {}; fr < 2 * radius + 1; fr++) {
        for (size_t fc {}; fc < 2 * radius + 1; fc++) {
            if ((threadX  + fr) >= radius && (threadX - radius + fr) < tile_width && (threadY + fc) >= radius && (threadY - radius + fc) < tile_width) {
                val += filter[fr * radius + fc] * N_s[threadX - radius + fr][threadY - radius + fc];
            } else if ((row + fr) >= radius && (row - radius + fr) < height && (col + fc) >= radius && (col - radius + fc) < width) {
                val += filter[fr * radius + fc] * N[(row - radius + fr) * width + col - radius + fc];
            }
        }
    }
    P[row * width + col] = val;
}

void conv2d(float* N, float* F, float* P, size_t radius, size_t width, size_t height)
{
    if (radius > PPMP_CONV2D_FILTER_MAX_RADIUS) {
        throw std::invalid_argument("Filter radius is too large.");
    }

    float* N_d = nullptr;
    float* P_d = nullptr;

    size_t filter_width = (2 * radius + 1);

    checkCudaError(cudaMalloc(&N_d, width * height * sizeof(float)));
    checkCudaError(cudaMalloc(&P_d, width * height * sizeof(float)));

    checkCudaError(cudaMemcpy(N_d, N, width * height * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpyToSymbol(filter, F, filter_width * filter_width * sizeof(float)));

    dim3 blockD = { PPMP_CONV2D_TILE_WIDTH, PPMP_CONV2D_TILE_WIDTH, 1 };
    dim3 gridD = { static_cast<unsigned int>(ceil(static_cast<float>(height) / PPMP_CONV2D_TILE_WIDTH)), static_cast<unsigned int>(ceil(static_cast<float>(width) / PPMP_CONV2D_TILE_WIDTH)), 1 };
    conv2dKernel<<<gridD, blockD>>>(N_d, P_d, radius, width, height);
    
    checkCudaLastError();

    checkCudaError(cudaMemcpy(P, P_d, width * height * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(N_d));
    checkCudaError(cudaFree(P_d));
}