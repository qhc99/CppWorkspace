# Requirements
- Supported environment: Clang/MSVC, Ubuntu 24 WSL2/Windows 11 (Visual Studio Community 2022), Cuda 12.
- Libraries: 
    1. `libomp-dev` (openmp, WSL2) 
    1. `llvm` (test coverage, WSL2)
    1. `vcpkg` (package manager)
    1. `ninja-build` (WSL2, optional by updating `CMakePresets.json`)
    1. `clangd` (optional to use clangd language server. It depends on `compile_commands.json`, which is currently only produced by `make` or `ninja`)
- Commands should be in path: 
    1. Compiler: `clang++`, `nvcc`
    1. Build: `vcpkg`, `cmake`, `ninja`, `msbuild` (multi-core build)
        - `msbuild` example path: `C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin`
    1. Tools: `clangd`, `llvm-cov`, `llvm-profdata`, `rg`(ripgrep)

# Usage
- Test coverage is generated to folder `_html_cov_report` by running corresponding custom targets.
- Run `run-checks.cmd` or `run-checks.sh` in project root dir to check if build and tests pass
    - `run-checks.cmd` cannot run multple times in one terminal because the env is polluted for clang
- Enable clangd cuda support on Linux: update the cuda path in `config.yaml` on your system and copy it to clangd user folder: ` cp config.yaml  ~/.config/clangd/config.yaml`. 
- Setup x64 env for ninja build in vscode on windows:
    - Add a new terminal profile and set it as default for `"terminal.integrated.profiles.windows"`:
    ```json
    "x64 Native Tools Command Prompt": {
        "path": "C:\\Windows\\System32\\cmd.exe",
        "args": [
            "/k",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"
        ]
    },
    ```
    - Note that this approach will break env variables for clang presets

# Troubleshooting
- On windows, Address sanitizer and Clang must be installed through Visual Studio to use ASan (official download dll crashes for some reason). 
    - ASan shared lib folder should be in the path (Example: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\lib\clang\17\lib\windows`, file `clang_rt.asan_dynamic-x86_64.dll` should be in the above folder)
- Correct architecture (`x86` vs `x64`)  of `clang++` should be used to avoid vcpkg error. 
- Build output folder: `_build`, which should match `.clangd` config. 

# Known limitations
1. `chip8` can only run in GUI environment.
1. On Windows, Cuda device code can only be debugged in Visual Studio (open `.sln` project file in `_msbuild` using preset `VS 2022`).
1. Test coverage targets are only generated if using Clang compiler.
1. CMake Clang support for cuda on windows is limited, so cuda programs will not be generated in this case. (Clangd is also not working since compilation commands are not exported)