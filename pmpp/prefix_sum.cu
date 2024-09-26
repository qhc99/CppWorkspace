#include "utils.h"
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
constexpr int KOGGE_STONE_SECTION_SIZE = 512;
constexpr int KOGGE_STONE_THREADS = KOGGE_STONE_SECTION_SIZE;
__global__ void KoggeStoneKernel(const float* data, float* out, unsigned int N)
{
    __shared__ float block_mem[KOGGE_STONE_SECTION_SIZE];
    unsigned int i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i < N) {
        block_mem[threadIdx.x] = data[i];
    } else {
        block_mem[threadIdx.x] = 0.F;
    }
    for (unsigned int stride { 1 }; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp {};
        if (threadIdx.x >= stride) {
            temp = block_mem[threadIdx.x] + block_mem[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            block_mem[threadIdx.x] = temp;
        }
    }
    if (i < N) {
        out[i] = block_mem[threadIdx.x];
    }
}
constexpr int BRENT_KUNG_THREADS = 512;
constexpr int BRENT_KUNG_SECTION_SIZE = BRENT_KUNG_THREADS * 2;

__global__ void BrentKungKernel(const float* data, float* out, unsigned int N)
{
    __shared__ float block_mem[BRENT_KUNG_SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        block_mem[threadIdx.x] = data[i];
    }
    if (i + blockDim.x < N) {
        block_mem[threadIdx.x + blockDim.x] = data[i + blockDim.x];
    }
    for (unsigned int stride { 1 }; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index { (threadIdx.x + 1) * 2 * stride - 1 };
        if (index < BRENT_KUNG_SECTION_SIZE) {
            block_mem[index] += block_mem[index - stride];
        }
    }
    for (int stride { BRENT_KUNG_SECTION_SIZE / 4 }; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index { (threadIdx.x + 1) * 2 * stride - 1 };
        if (index + stride < BRENT_KUNG_SECTION_SIZE) {
            block_mem[index + stride] += block_mem[index];
        }
    }
    __syncthreads();
    if (i < N) {
        out[i] = block_mem[threadIdx.x];
    }
    if (i + blockDim.x < N) {
        out[i + blockDim.x] = block_mem[threadIdx.x + blockDim.x];
    }
}

void KoggeStoneSegmentScan(float* data, float* out, unsigned int length)
{
    float* data_d {};
    float* out_d {};
    checkCudaError(cudaMalloc(&data_d, length * sizeof(float)));
    checkCudaError(cudaMalloc(&out_d, length * sizeof(float)));
    checkCudaError(cudaMemcpy(data_d, data, length * sizeof(float), cudaMemcpyHostToDevice));

    KoggeStoneKernel<<<static_cast<unsigned int>(ceil(static_cast<double>(length) / static_cast<double>(KOGGE_STONE_SECTION_SIZE))), KOGGE_STONE_THREADS>>>(data_d, out_d, length);

    checkCudaLastError();

    checkCudaError(cudaMemcpy(out, out_d, length * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(data_d));
    checkCudaError(cudaFree(out_d));
}

void BrentKungSegmentScan(float* data, float* out, unsigned int length)
{
    float* data_d {};
    float* out_d {};
    checkCudaError(cudaMalloc(&data_d, length * sizeof(float)));
    checkCudaError(cudaMalloc(&out_d, length * sizeof(float)));
    checkCudaError(cudaMemcpy(data_d, data, length * sizeof(float), cudaMemcpyHostToDevice));

    BrentKungKernel<<<static_cast<unsigned int>(ceil(static_cast<double>(length) / static_cast<double>(BRENT_KUNG_SECTION_SIZE))), BRENT_KUNG_THREADS>>>(data_d, out_d, length);

    checkCudaLastError();

    checkCudaError(cudaMemcpy(out, out_d, length * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(data_d));
    checkCudaError(cudaFree(out_d));
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
