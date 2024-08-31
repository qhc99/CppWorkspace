#ifndef PMPP_UTILS_H
#define PMPP_UTILS_H

#include "workspace_pch.h"

#define checkCudaError(val) check_cuda_error( (val), #val, __FILE__, __LINE__ )
#define checkCudaLastError() check_cuda_last_error(__FILE__, __LINE__ )

inline void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != 0) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "; " <<  cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        throw std::runtime_error("Exit");
    }
}

inline void check_cuda_last_error(const char *const file, int const line){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << ", " << "file: " << file << "line: " << line;
    }
}
#endif