cmake_minimum_required(VERSION 3.28)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    project(opengl_learn LANGUAGES CXX)

    find_package(OpenGL REQUIRED COMPONENTS OpenGL)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(Stb REQUIRED)
    find_package(glm CONFIG REQUIRED)

    add_executable(wireframe_ogl demos/wireframe.cpp)
    set_target_properties(wireframe_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(wireframe_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(system_check_ogl demos/system_check.cpp)
    set_target_properties(system_check_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(system_check_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(texture_box_anime_ogl demos/texture_box_anime.cpp includes/shader_s.h)
    set_target_properties(texture_box_anime_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(texture_box_anime_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(texture_box_anime_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(texture_box_anime_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(texture_box_anime_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(texture_box_anime_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/wall.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/wall.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.3.coordinate_systems.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.3.coordinate_systems.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.3.coordinate_systems.fs
        ${CMAKE_CURRENT_BINARY_DIR}/6.3.coordinate_systems.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/awesomeface.png
        ${CMAKE_CURRENT_BINARY_DIR}/awesomeface.png
        COPYONLY)
endif()