cmake --preset "Clang Debug"
cd _build_debug
cmake --install . --prefix ../standalone_import/_installed
cmake --build . --target clean

echo "[$(date '+%H:%M:%S')]: Build start"
cmake --build . --target all  --parallel
echo "[$(date '+%H:%M:%S')]: Build end"

ctest 
cmake --build . --target clean
cd ..
rd -r _installed