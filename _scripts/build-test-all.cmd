cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target clean

call "%~dp0\print_clock.cmd" "Build start"
cmake --build . --target all --parallel
call "%~dp0\print_clock.cmd" "Build end"


ctest
rm -rf ../archieve/standalone_import/_installed
cmake --install . --prefix ../archieve/standalone_import/_installed
cmake --build . --target clean
cd ..

cmake --preset "VS 2022"
cd _msbuild
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug

call "%~dp0\print_clock.cmd" "Build start"
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Build" /property:Configuration=Debug
call "%~dp0\print_clock.cmd" "Build end"

ctest -C Debug
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug
cd ..

cd archieve/standalone_import
cmake --preset "Clang Debug"
cd _build_debug
cmake --build . --target all --parallel
ctest
cmake --build . --target clean
cd ../..