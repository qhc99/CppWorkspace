#ifndef PMPP_EXAMPLE_UTILS
#define PMPP_EXAMPLE_UTILS

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != 0) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "; " <<  cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
#endif