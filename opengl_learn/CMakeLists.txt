cmake_minimum_required(VERSION 3.28)

project(opengl_learn)

find_package(OpenGL REQUIRED)
target_link_libraries(main PRIVATE OpenGL::GL)