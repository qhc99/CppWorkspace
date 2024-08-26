@echo off
echo The current time is: %time:~0,8%
echo on

cmake --preset "Clang Release"
cd _build
cmake --build . --target clean
cmake --build . --target all
ctest -E ^asan_.*
cd ..

@echo off
echo The current time is: %time:~0,8%
echo on

cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target clean
cmake --build . --target all
ctest -E ^asan_.*
cd ..

@echo off
echo The current time is: %time:~0,8%
echo on 

cmake --preset "VS 2022"
cd _msbuild
cmake --build . --target ALL_BUILD --config Debug
ctest -C Debug -E ^asan_.*

@echo off
echo The current time is: %time:~0,8%
echo on 

cmake --build . --target ALL_BUILD --config Release
ctest -C Debug -E ^asan_.*  
cd ..

@echo off
echo The current time is: %time:~0,8%
echo on
