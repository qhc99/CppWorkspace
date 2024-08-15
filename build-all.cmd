cmake --preset "Clang Release"
cd _build
cmake --build . --target all 
cd ..

cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all 
cd ..

cmake --preset "VS 2022"
cd _msbuild
cmake --build . --target ALL_BUILD --config Release
cmake --build . --target ALL_BUILD --config Debug
cd ..