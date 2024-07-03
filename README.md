# CMakeList requirements

- All projects are developed by Clang.
- Set env var `CUDACXX` to use NVCC under WSL.
- Test coverage is generated to `./html_cov_report` by running corresponding custom target.
- Required libraries: 
    1. `libomp-dev` (ray trace openmp) 
    1. `llvm` (test coverages)
    1. `vcpkg`
    1. `ninja-build`
- Build target `doctest` before building all or test targets