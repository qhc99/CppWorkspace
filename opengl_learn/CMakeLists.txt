cmake_minimum_required(VERSION 3.28)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    project(opengl_learn LANGUAGES CXX)

    find_package(OpenGL REQUIRED COMPONENTS OpenGL) # v3.10
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)

    add_executable(triangle_ogl triangle.cpp)
    set_target_properties(triangle_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(triangle_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(rectangle_ogl rectangle.cpp)
    set_target_properties(rectangle_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(rectangle_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(wireframe_ogl wireframe.cpp)
    set_target_properties(wireframe_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(wireframe_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(system_check_ogl system_check.cpp)
    set_target_properties(system_check_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(system_check_ogl PRIVATE OpenGL::GL glad::glad glfw)
endif()