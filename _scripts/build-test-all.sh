cmake --preset "Clang Release"
cd _build
cmake --build . --target all
ctest -E ^asan_.*
cd ..

cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all
ctest -E ^asan_.*
cd ..