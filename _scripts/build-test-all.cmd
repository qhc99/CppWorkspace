cmake --preset "Clang Release"
cd _build
cmake --build . --target clean
cmake --build . --target all
ctest -E ^asan_.*
cd ..

cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target clean
cmake --build . --target all
ctest -E ^asan_.*
cd ..

cmake --preset "VS 2022"
cd _msbuild
cmake --build . --target clean
cmake --build . --target ALL_BUILD --config Release
ctest -C Release -E ^asan_.* 
cmake --build . --target ALL_BUILD --config Debug
ctest -C Debug -E ^asan_.* 
cd ..
