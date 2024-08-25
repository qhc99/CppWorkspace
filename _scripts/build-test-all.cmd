date
cmake --preset "Clang Release"
cd _build
cmake --build . --target all
ctest -E ^asan_.*
cd ..

date

cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all
ctest -E ^asan_.*
cd ..

date 

cmake --preset "VS 2022"
cd _msbuild
cmake --build . --target ALL_BUILD --config Debug
ctest -C Debug -E ^asan_.*

date 

cmake --build . --target ALL_BUILD --config Release
ctest -C Debug -E ^asan_.*  
cd ..

date
