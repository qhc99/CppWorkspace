# CMakeList requirements

- Environment: Clang, Linux(WSL2).
- Set env var `CUDACXX` to use NVCC.
- Test coverage is generated to folder `_html_cov_report` by running corresponding custom targets.
- Libraries: 
    1. `libomp-dev` (ray trace openmp) 
    1. `llvm` (test coverages)
    1. `vcpkg` (package manager)
    1. `ninja-build` (optional by `CMakePresets.json`)
    1. `clangd` (optional to use clang language server)
- Build target `doctest` before building all test targets
- Build output folder: `_build`, which should match `.clangd` config. 