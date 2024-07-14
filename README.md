# CMakeList requirements

- Environment: Clang, Linux.
- Set env var `CUDACXX` to use NVCC.
- Test coverage is generated to folder `_html_cov_report` by running corresponding custom targets.
- Libraries: 
    1. `libomp-dev` (ray trace openmp) 
    1. `llvm` (test coverages)
    1. `vcpkg`
    1. `ninja-build` (optional by modifying presets)
    1. `clangd` (optional to use clang language server)
- Build target `doctest`: `cmake --build build -t doctest` before building all or test targets
- Build output folder: `_build`, which should match `.clangd` config. Build output should stay in a single folder to use compilation database for `clangd`.