# Requirements
- Supported environment: Clang/MSVC, Ubuntu 24 WSL2/Windows 11 (Visual Studio Community 2022), Cuda 12.
- Libraries: 
    1. `libomp-dev` (openmp, WSL2) 
    1. `llvm` (test coverage, WSL2)
    1. `vcpkg` (package manager)
    1. `ninja-build` (WSL2, optional by updating `CMakePresets.json`)
    1. `clangd` (optional to use clangd language server. It depends on `compile_commands.json`, which is currently only produced by `make` or `ninja`)
- Commands should be in path: 
    1. Compiler: `clang`, `clang++`, `nvcc`, `cl`
    1. Build: `vcpkg`, `cmake`, `ninja`, `mt` (windows SDK used by CUDA, example path: `C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64`)
    1. Tools: `clangd`, `llvm-cov`, `llvm-profdata`

# Usage
- Test coverage is generated to folder `_html_cov_report` by running corresponding custom targets.
- Run `build-all.cmd` or `build-all.sh` to check if buildings pass

# Troubleshooting
- On windows, Address sanitizer and Clang must be installed through Visual Studio to use ASan (official download dll crashes for some reason). 
    - ASan shared lib folder should be in the path (Example: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\lib\clang\17\lib\windows`, file `clang_rt.asan_dynamic-x86_64.dll` should be in the above folder)
- Correct architecture (`x86` vs `x64`)  of `clang++` should be used to avoid vcpkg error. 
- Build output folder: `_build`, which should match `.clangd` config. 

# Known limitations
1. `chip8` can only run in GUI environment.
1. On Windows, Cuda device code can only be debugged in Visual Studio (open `.sln` project file in `_msbuild` using preset `VS 2022`).
1. Test coverage targets are only generated if using Clang compiler.
1. Clangd does not have complete support for cuda.
1. Clang has limited supported for cuda on windows, so cuda programs will not be generated in this case.