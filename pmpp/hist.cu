#define PPMP_CONV2D_FILTER_MAX_RADIUS 5
#define PPMP_CONV2D_TILE_WIDTH 32

#include "utils.h"
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)

__global__ void histKernel(char* data, unsigned int length, unsigned int hist, unsigned int bin_count)
{
    extern __shared__ unsigned int hist_shared[];
    for (unsigned int bin_idx = threadIdx.x; bin_idx < bin_count; bin_idx += blockDim.x) {
        hist_shared[bin_idx] = 0U;
    }
    __syncthreads();

}

void hist(char* data, unsigned int length, unsigned int hist, unsigned int bin_count)
{
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
