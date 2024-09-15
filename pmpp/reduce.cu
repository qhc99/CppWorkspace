#include "utils.h"

#ifndef PMPP_REDUCE_KERNEL_COARSE_FACTOR
#define PMPP_REDUCE_KERNEL_COARSE_FACTOR 2
#endif

#define PMPP_REDUCE_KERNEL_BLOCK_DIM 128

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
template <typename T>
struct MaxOp {
    __device__ T operator()(T a, T b) const
    {
        return a > b ? a : b;
    }

    __device__ T identity() const
    {
        return std::numeric_limits<T>::lowest();
    }
};

template <typename T>
struct AtomicMaxOp {
    __device__ void operator()(T* a, T b) const
    {
        atomicMax(a, b);
    }
};

template <typename T>
struct MinOp {
    __device__ T operator()(T a, T b) const
    {
        return a < b ? a : b;
    }

    __device__ T identity() const
    {
        return std::numeric_limits<T>::max();
    }
};

template <typename T>
struct AtomicMinOp {
    __device__ void operator()(T* a, T b) const
    {
        atomicMin(a, b);
    }
};

template <typename T>
struct AddOp {
    __device__ T operator()(T a, T b) const
    {
        return a + b;
    }

    __device__ T identity() const
    {
        return T(0);
    }
};

template <typename T>
struct AtomicAddOp {
    __device__ void operator()(T* a, T b) const
    {
        atomicAdd(a, b);
    }
};

template <typename T, typename Op, typename AtomicOp>
__global__ void reduceKernel(const T* input, size_t length, T* output, Op reduce_func, AtomicOp atomic_reduce_func)
{
    __shared__ T input_s[PMPP_REDUCE_KERNEL_BLOCK_DIM];
    unsigned int segment { PMPP_REDUCE_KERNEL_COARSE_FACTOR * 2 * blockDim.x * blockIdx.x };
    unsigned int i { segment + threadIdx.x };
    unsigned int t { threadIdx.x };

    input_s[t] = reduce_func.identity();
    __syncthreads();

    if (i < length) {
        T val = input[i];
        for (unsigned int tile { 1 };
             (tile < PMPP_REDUCE_KERNEL_COARSE_FACTOR * 2) && (i + tile * PMPP_REDUCE_KERNEL_BLOCK_DIM < length);
             ++tile) {
            val = reduce_func(val, input[i + tile * PMPP_REDUCE_KERNEL_BLOCK_DIM]);
        }
        input_s[t] = val;
    }

    for (unsigned int stride { blockDim.x / 2 }; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] = reduce_func(input_s[t], input_s[t + stride]);
        }
    }
    if (t == 0) {
        atomic_reduce_func(output, input_s[0]);
    }
}

/**
 * @brief
 *
 * @tparam T
 * @tparam Op (T,T)->T
 * @param input
 * @param length
 * @param output
 */
template <typename T, typename Op, typename AtomicOp>
void reduce(const T* input, size_t length, T* output, Op reduce_func, AtomicOp atomic_reduce_func)
{

    T* input_d = nullptr;
    T* output_d = nullptr;

    checkCudaError(cudaMalloc(&input_d, length * sizeof(T)));
    checkCudaError(cudaMalloc(&output_d, sizeof(T)));

    checkCudaError(cudaMemcpy(input_d, input, length * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(output_d, output, sizeof(T), cudaMemcpyHostToDevice));
    reduceKernel<T, Op, AtomicOp><<<8, PMPP_REDUCE_KERNEL_BLOCK_DIM>>>(input_d, length, output_d, reduce_func, atomic_reduce_func);
    checkCudaLastError();

    checkCudaError(cudaMemcpy(output, output_d, sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(input_d));
    checkCudaError(cudaFree(output_d));
}

void reduce_min_i(const int* input, size_t length, int* output)
{
    reduce(input, length, output, MinOp<int> {}, AtomicMinOp<int> {});
}

void reduce_max_i(const int* input, size_t length, int* output)
{
    reduce(input, length, output, MaxOp<int> {}, AtomicMaxOp<int> {});
}

void reduce_add_f(const float* input, size_t length, float* output)
{
    reduce(input, length, output, AddOp<float> {}, AtomicAddOp<float> {});
}

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
