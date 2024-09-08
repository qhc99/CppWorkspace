cmake --preset "Clang Debug"
cd _build_debug

@echo off
echo Build start, the current time is: %time:~0,8%
echo on
cmake --build . --target all --parallel
@echo off
echo Build end, the current time is: %time:~0,8%
echo on

ctest -E ^asan_.*
cmake --build . --target clean
cd ..

cmake --preset "VS 2022"
cd _msbuild

@echo off
echo Build start, the current time is: %time:~0,8%
echo on 
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Build" /property:Configuration=Debug
@echo off
echo Build end, the current time is: %time:~0,8%
echo on

ctest -C Debug -E ^asan_.*
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug
cd ..