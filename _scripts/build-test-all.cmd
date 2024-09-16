cmake --preset "Clang Debug"
cd _build_debug

call "%~dp0\print_clock.cmd"
cmake --build . --target all --parallel
call "%~dp0\print_clock.cmd"

ctest -E ^asan_.*
cmake --build . --target clean
cd ..

cmake --preset "VS 2022"
cd _msbuild

call "%~dp0\print_clock.cmd"
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Build" /property:Configuration=Debug
call "%~dp0\print_clock.cmd"

ctest -C Debug -E ^asan_.*
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug
cd ..