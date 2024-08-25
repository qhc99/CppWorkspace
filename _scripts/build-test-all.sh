cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all
ctest -E ^asan_.*
cd ..