# Requirements
- Environment: Clang 18, Ubuntu 24 WSL2/Windows 11, Cuda 12, (Visual Studio 2022).
- Libraries: 
    1. `libomp-dev` (ray trace openmp, WSL2) 
    1. `llvm` (test coverage)
    1. `vcpkg` (package manager)
    1. `ninja-build` (optional by updating `CMakePresets.json`)
    1. `clangd` (optional to use clang language server)

# Usage
- Build target `doctest` before building all test targets
- Test coverage is generated to folder `_html_cov_report` by running corresponding custom targets.


# Troubleshooting
- CUDA: set env var `CUDACXX` to use NVCC.
- Address sanitizer: need to install through Visual Studio
- Command exist in path: 
    1. `clang`
    1. `clang++`
    1. `ninja`
    1. `cmake`
    1. `nvcc`
    1. `vcpkg`
    1. `clangd`
    1. `mt` (CUDA on windows)
- Build output folder: `_build`, which should match `.clangd` config. 
