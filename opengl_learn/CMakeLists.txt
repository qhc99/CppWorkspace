cmake_minimum_required(VERSION 3.28)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    project(opengl_learn LANGUAGES CXX)

    find_package(OpenGL REQUIRED COMPONENTS OpenGL)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(Stb REQUIRED)

    add_executable(triangle_ogl demos/triangle.cpp)
    set_target_properties(triangle_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(triangle_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(rectangle_ogl demos/rectangle.cpp)
    set_target_properties(rectangle_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(rectangle_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(wireframe_ogl demos/wireframe.cpp)
    set_target_properties(wireframe_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(wireframe_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(system_check_ogl demos/system_check.cpp)
    set_target_properties(system_check_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(system_check_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(uniform_ogl demos/uniform.cpp)
    set_target_properties(uniform_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(uniform_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(vert_attr_ogl demos/vert_attr.cpp)
    set_target_properties(vert_attr_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(vert_attr_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(texture_ogl demos/texture.cpp includes/shader_s.h)
    set_target_properties(texture_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(texture_ogl PRIVATE OpenGL::GL glad::glad glfw)
    target_include_directories(texture_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(texture_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(texture_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/wall.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/demos/wall.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.texture.vs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.texture.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.texture.fs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.texture.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/awesomeface.png
        ${CMAKE_CURRENT_BINARY_DIR}/demos/awesomeface.png
        COPYONLY)
endif()