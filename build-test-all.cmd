git pull

cmake --preset "Clang Release"
cd _build
cmake --build . --target all
ctest -V -E ^asan_.*
cd ..

cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all
ctest -V -E ^asan_.*
cd ..

cmake --preset "VS 2022"
cd _msbuild
cmake --build . --target ALL_BUILD --config Release
ctest -V -C Release -E ^asan_.* 
cmake --build . --target ALL_BUILD --config Debug
ctest -V -C Debug -E ^asan_.* 
cd ..