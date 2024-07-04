# CMakeList requirements

- All projects are developed by Clang.
- Set env var `CUDACXX` to use NVCC under WSL.
- Test coverage is generated to `./html_cov_report` by running corresponding custom target.
- Libraries: 
    1. `libomp-dev` (ray trace openmp) 
    1. `llvm` (test coverages)
    1. `vcpkg`
    1. `ninja-build` (optional by modifying presets)
    1. `clangd` (optional to use clang language server)
- Generate cmake build folder: `cmake --preset=vcpkg`
- Build target `doctest`: `cmake --build build -t doctest` before building all or test targets
- Build output folder: `_build`, which should match `.clangd` config