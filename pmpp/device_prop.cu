#include "workspace_pch.h"
#include "utils.h"

int main()
{
    int devCount{};
    checkCudaErrors(cudaGetDeviceCount(&devCount));

    cudaDeviceProp devProp{};
    for (int i = 0; i < devCount; i++) {
        checkCudaErrors(cudaGetDeviceProperties(&devProp, i));
        std::cout << devProp.name << "\n";
        std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << "\n";
        std::cout << "multiProcessorCount(SM): " << devProp.multiProcessorCount << "\n";
        std::cout << "maxThreadsDim[0]: " << devProp.maxThreadsDim[0] << "\n";
        std::cout << "maxThreadsDim[1]: " << devProp.maxThreadsDim[1] << "\n";
        std::cout << "maxThreadsDim[2]: " << devProp.maxThreadsDim[2] << "\n";
        std::cout << "maxGridSize[0]: " << devProp.maxGridSize[0] << "\n";
        std::cout << "maxGridSize[1]: " << devProp.maxGridSize[1] << "\n";
        std::cout << "maxGridSize[2]: " << devProp.maxGridSize[2] << "\n";
        std::cout << "regsPerBlock: " << devProp.regsPerBlock << "\n";
        std::cout << "warpSize: " << devProp.warpSize << "\n";
        std::cout << "\n";
    }

    return 0;
}