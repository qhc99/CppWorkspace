cmake --preset "Clang Debug"
cd _build_debug
cmake --install . --prefix ../standalone_import/_installed
cmake --build . --target clean

call "%~dp0\print_clock.cmd" "Build start"
cmake --build . --target all --parallel
call "%~dp0\print_clock.cmd" "Build end"

ctest
cmake --build . --target clean
cd ..

cmake --preset "VS 2022"
cd _msbuild
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug

call "%~dp0\print_clock.cmd" "Build start"
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Build" /property:Configuration=Debug
call "%~dp0\print_clock.cmd" "Build end"

ctest
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug
cd ..
rd -r _installed