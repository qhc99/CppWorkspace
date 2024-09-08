cmake_minimum_required(VERSION 3.28)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    project(opengl_learn LANGUAGES CXX)

    find_package(OpenGL REQUIRED COMPONENTS OpenGL) # v3.10
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    add_executable(windows_ogl window.cpp)
    set_target_properties(windows_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(windows_ogl PRIVATE OpenGL::GL glad::glad glfw)
endif()