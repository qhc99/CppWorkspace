cmake_minimum_required (VERSION 3.17)

project ("cuda_demos" LANGUAGES CUDA)
include(CheckLanguage)
check_language(CUDA)

add_executable (cuda_hello_world hello_world.cu)
target_include_directories(cuda_hello_world PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_hello_world PRIVATE ${CMAKE_CUDA_RUNTIME_LIBRARY} )
set_property(TARGET cuda_hello_world PROPERTY CUDA_STANDARD 17 )
set_property(TARGET cuda_hello_world PROPERTY CUDA_ARCHITECTURES "native")


