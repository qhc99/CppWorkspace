# CMakeList :

All projects are developed using Clang.

Set env var `CUDACXX` to use NVCC preset under WSL.

# Test Coverage Example

```
llvm-profdata merge -sparse default.profraw -o foo.profdata

llvm-cov show -format=html -o html_cov_report ./a.exe -instr-profile="foo.profdata"
```