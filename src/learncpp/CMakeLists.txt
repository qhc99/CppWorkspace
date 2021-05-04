cmake_minimum_required(VERSION 3.16)

project(learncpp)

add_executable(main.exe main.cpp)

target_link_libraries(main.exe
        org::nathan::utils
        org::nathan::demos
        )

#    message(WARNING "Hardcode clang-cl address sanitizer")
#    target_compile_options(main.exe PRIVATE -fsanitize=address)
#    target_link_directories(main.exe
#            PRIVATE
#            "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/2019/Community/VC/Tools/Llvm/x64/lib/clang/11.0.0/lib/windows")
#    target_link_libraries(main.exe PRIVATE
#            clang_rt.asan_dynamic-x86_64
#            clang_rt.asan_dynamic_runtime_thunk-x86_64
#            org::nathan::utils
#            org::nathan::demos
#            )
#    target_link_options(main.exe PRIVATE /wholearchive:clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)


