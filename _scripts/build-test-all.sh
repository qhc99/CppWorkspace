cmake --preset "Clang Debug"
cd _build_debug

echo "[$(date '+%H:%M:%S')]: Build start"
cmake --build . --target all  --parallel
echo "[$(date '+%H:%M:%S')]: Build end"

ctest
cmake --install . --prefix ../standalone_import/_installed
cmake --build . --target clean
cd ..