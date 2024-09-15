#ifndef PPMP_CONV2D_FILTER_MAX_RADIUS
#define PPMP_CONV2D_FILTER_MAX_RADIUS 5
#endif

#ifndef PPMP_CONV2D_TILE_WIDTH
#define PPMP_CONV2D_TILE_WIDTH 32
#endif

#include "utils.h"
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)

__global__ void histKernel(const char* data, unsigned int length, unsigned int* hist_out, unsigned int bin_width, bool use_aggregation)
{
    extern __shared__ unsigned int hist_shared[]; // bin_count * sizeof(unsigned int)
    const unsigned int bin_count = ceil(26. / bin_width);
    for (unsigned int bin_idx = threadIdx.x; bin_idx < bin_count; bin_idx += blockDim.x) {
        hist_shared[bin_idx] = 0U;
    }
    __syncthreads();
    if (!use_aggregation) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (auto i { tid }; i < length; i += blockDim.x * gridDim.x) {
            int c { data[i] - 'a' };
            if (c >= 0 && c < 26) {
                atomicAdd(&(hist_shared[c / bin_width]), 1);
            }
        }
    } else {
        unsigned int acc {};
        int prev_bin_idx { -1 };
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (auto i { tid }; i < length; i += blockDim.x * gridDim.x) {
            int c { data[i] - 'a' };
            if (c >= 0 && c < 26) {
                int bin_idx = c / static_cast<int>(bin_width);
                if (bin_idx == prev_bin_idx) {
                    ++acc;
                } else {
                    if (acc > 0) {
                        atomicAdd(&(hist_shared[prev_bin_idx]), acc);
                    }
                    acc = 1;
                    prev_bin_idx = bin_idx;
                }
            }
        }
        if (acc > 0) {
            atomicAdd(&(hist_shared[prev_bin_idx]), acc);
        }
    }
    __syncthreads();
    for (unsigned int bin_idx { threadIdx.x }; bin_idx < bin_count; bin_idx += blockDim.x) {
        auto v { hist_shared[bin_idx] };
        if (v > 0) {
            atomicAdd(&(hist_out[bin_idx]), v);
        }
    }
}

void hist(char* data, unsigned int length, unsigned int* hist_out, unsigned int bin_width, bool use_aggregation)
{
    char* data_d = nullptr;
    unsigned int* hist_d = nullptr;
    const auto bin_count = static_cast<unsigned int>(ceil(26. / bin_width));
    checkCudaError(cudaMalloc(&data_d, length * sizeof(char)));
    checkCudaError(cudaMalloc(&hist_d, bin_count * sizeof(unsigned int)));
    checkCudaError(cudaMemcpy(data_d, data, length * sizeof(char), cudaMemcpyHostToDevice));

    histKernel<<<4, 256, bin_count * sizeof(unsigned int)>>>(data_d, length, hist_d, bin_width, use_aggregation);
    
    checkCudaLastError();

    checkCudaError(cudaMemcpy(hist_out, hist_d, bin_count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(data_d));
    checkCudaError(cudaFree(hist_d));
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
