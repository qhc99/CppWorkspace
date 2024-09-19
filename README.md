# Requirements
- Supported environment: 
    - Ubuntu 24 WSL2/Windows 11 (VS 2022)
    - Clang-Ninja-Win/Linux, 
    - MSVC-MSBuild
    - Cuda 12
    - All cmake presets work on windows, only clang presets work on linux.
- Libraries: 
    1. `libomp-dev` (openmp, WSL2) 
    1. `llvm` (test coverage, WSL2)
    1. `vcpkg` (package manager)
    1. `ninja-build` (optional on windows)
    1. `clangd` (optional to use clangd language server. It depends on `compile_commands.json`, which is currently only produced by `make` or `ninja`)
- Commands should be in path: 
    1. Compilers: `clang++`, `nvcc`
    1. Build: `vcpkg`, `cmake`, `ninja`, `msbuild`
        - `msbuild` example path: `C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin`
    1. Tools: `clangd`, `llvm-cov`, `llvm-profdata`, `rg`(ripgrep)

# Usage
- Test coverage is generated to folder `_html_cov_report` by running corresponding custom targets.
- Run `run-checks.cmd` or `run-checks.sh` in project root dir to check if build and tests pass
    - `run-checks.cmd` cannot run multple times in one env because the msvc env will polluted clang asan env
- Enable clangd cuda support on Linux: update the cuda path in `config.yaml` on your system and copy it to clangd user folder: `cp ./scripts/config.yaml  ~/.config/clangd/config.yaml`. 

# Troubleshooting
- On windows, Address sanitizer and Clang must be installed through Visual Studio to use ASan (official download dll crashes for some reason). 
    - ASan shared lib folder should be in the path (Example: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\lib\clang\17\lib\windows`, file `clang_rt.asan_dynamic-x86_64.dll` should be in the above folder)
- Correct architecture (`x86` vs `x64`)  of `clang++` should be used to avoid vcpkg error. 
- Build output folder: `_build_debug`, which should match `.clangd` config. 

# Known limitations
1. GUI projects are not supported on WSL2.
1. On Windows, Cuda device code can only be debugged in Visual Studio (open `.sln` project file in `_msbuild` using preset `VS 2022`).
1. Test coverage targets are only generated if using Clang compiler.
1. Clang support for cuda on windows is limited, so cuda programs will not be generated in this case. (Clangd is also not working since compilation commands are not exported)
1. Ninja does not bring noticeable performance improvement than msbuild on windows and it requires to manually
call `vcvars64.bat` to setup essential env variables (version update will break manually path update), so ninja+msvc is not supported in this project.