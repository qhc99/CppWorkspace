cmake --preset "Clang Debug"
rd -r _installed
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
rd -r _installed
cd _msbuild
cmake --install . --prefix ../standalone_import/_installed
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug

call "%~dp0\print_clock.cmd" "Build start"
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Build" /property:Configuration=Debug
call "%~dp0\print_clock.cmd" "Build end"

ctest
msbuild /m /verbosity:minimal ALL_BUILD.vcxproj /target:"Clean" /property:Configuration=Debug
cd ..
rd -r _installed