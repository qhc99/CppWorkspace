# Requirements
- Environment: Clang 17, Ubuntu 24 WSL2/Windows 11 (Required Visual Studio Community 2022), Cuda 12.
- Libraries: 
    1. `libomp-dev` (ray trace openmp, WSL2) 
    1. `llvm` (test coverage)
    1. `vcpkg` (package manager)
    1. `ninja-build` (optional by updating `CMakePresets.json`)
    1. `clangd` (optional to use clangd language server. It depends on `compile_commands.json`, which is currently only produced by `make` or `ninja`)

# Usage
- Build target `doctest` before building other test targets
- Test coverage is generated to folder `_html_cov_report` by running corresponding custom targets.


# Troubleshooting
- CUDA: set env var `CUDACXX` path to `nvcc`. 
    1. Cuda toolkit installation should add `CUDA_PATH` and `CUDA_PATH_${version}` to env. 
    1. Folders path `${cuda install folder}/${version}/bin` and `${cuda install folder}/${version}/libnvpp` should also be added to path on installation.
- Address sanitizer and Clang (LLVM) should be installed through Visual Studio on windows. 
    1. ASan shared lib folder should be in the path (Example: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64`).
    1. Check if `clang_rt.asan_dynamic-x86_64.dll` is in the above folder
- Commands should exist in path: 
    1. `clang`, `clang++`, `nvcc`, `cl` (CUDA on windows)
    1. `cmake`, `ninja`, `mt` (CUDA on windows)
    1. `clangd`, `llvm-cov`, `llvm-profdata`, `vcpkg`
- Check if is using the correct architecture of `clang++` (`x86` vs `x64`) to avoid vcpkg error. 
- Build output folder: `_build`, which should match `.clangd` config. 
