date

cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all  --parallel
ctest -E ^asan_.*
cd ..

date