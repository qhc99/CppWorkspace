cmake --preset "Clang Debug"
cd _build_debug

echo "[$(date '+%H:%M:%S')]: Build start"
cmake --build . --target all  --parallel
echo "[$(date '+%H:%M:%S')]: Build end"

ctest
rm -rf ../standalone_import/_installed
cmake --install . --prefix ../standalone_import/_installed
cmake --build . --target clean
cd ..

cd standalone_import
cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all --parallel
ctest
cmake --build . --target clean
cd ../..