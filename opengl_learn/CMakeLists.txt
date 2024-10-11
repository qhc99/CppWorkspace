cmake_minimum_required(VERSION 3.28)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    project(opengl_learn LANGUAGES CXX)

    find_package(OpenGL REQUIRED COMPONENTS OpenGL)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(Stb REQUIRED)
    find_package(glm CONFIG REQUIRED)
    find_package(assimp CONFIG REQUIRED)

    add_executable(wireframe_ogl demos/wireframe.cpp)
    set_target_properties(wireframe_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(wireframe_ogl PRIVATE OpenGL::GL glad::glad glfw)

    add_executable(system_check_ogl demos/system_check.cpp)
    set_target_properties(system_check_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(system_check_ogl PRIVATE OpenGL::GL glad::glad glfw)


    add_executable(phong_light_casters_ogl demos/phong_light_casters.cpp includes/shader_m.h includes/camera.h)
    set_target_properties(phong_light_casters_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(phong_light_casters_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(phong_light_casters_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(phong_light_casters_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(phong_light_casters_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(phong_light_casters_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.light_cube.vs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.light_cube.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.light_cube.fs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.light_cube.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.multiple_lights.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.multiple_lights.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.multiple_lights.fs
        ${CMAKE_CURRENT_BINARY_DIR}/6.multiple_lights.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/container2_specular.png
        ${CMAKE_CURRENT_BINARY_DIR}/container2_specular.png
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/container2.png
        ${CMAKE_CURRENT_BINARY_DIR}/container2.png
        COPYONLY)

    add_executable(stencil_outline_ogl demos/stencil_outline.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(stencil_outline_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(stencil_outline_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(stencil_outline_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(stencil_outline_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(stencil_outline_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(stencil_outline_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/2.stencil_testing.vs
        ${CMAKE_CURRENT_BINARY_DIR}/2.stencil_testing.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/2.stencil_testing.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.stencil_testing.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/2.stencil_single_color.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.stencil_single_color.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/marble.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/marble.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/metal.png
        ${CMAKE_CURRENT_BINARY_DIR}/metal.png
        COPYONLY)

    add_executable(blend_transparent_ogl demos/blend_transparent.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(blend_transparent_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(blend_transparent_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(blend_transparent_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(blend_transparent_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(blend_transparent_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(blend_transparent_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/3.2.blending.vs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.blending.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/3.2.blending.fs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.blending.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/marble.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/marble.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/metal.png
        ${CMAKE_CURRENT_BINARY_DIR}/metal.png
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/window.png
        ${CMAKE_CURRENT_BINARY_DIR}/window.png
        COPYONLY)

    add_executable(framebuffer_rearview_ogl demos/framebuffer_rearview.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(framebuffer_rearview_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(framebuffer_rearview_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(framebuffer_rearview_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(framebuffer_rearview_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(framebuffer_rearview_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(framebuffer_rearview_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/5.2.framebuffers.vs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/5.2.framebuffers.fs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/5.2.framebuffers_screen.vs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers_screen.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/5.2.framebuffers_screen.fs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers_screen.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/container.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/container.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/metal.png
        ${CMAKE_CURRENT_BINARY_DIR}/metal.png
        COPYONLY)

    add_executable(skybox_ogl demos/skybox.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(skybox_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(skybox_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(skybox_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(skybox_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(skybox_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(skybox_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.2.cubemaps.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.2.cubemaps.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.2.cubemaps.fs
        ${CMAKE_CURRENT_BINARY_DIR}/6.2.cubemaps.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.2.skybox.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.2.skybox.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.2.skybox.fs
        ${CMAKE_CURRENT_BINARY_DIR}/6.2.skybox.fs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/skybox
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(geometry_shader_normal_ogl demos/geometry_shader_normal.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(geometry_shader_normal_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(geometry_shader_normal_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(geometry_shader_normal_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(geometry_shader_normal_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(geometry_shader_normal_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(geometry_shader_normal_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/9.3.default.vs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.default.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/9.3.default.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.default.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/9.3.normal_visualization.vs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.normal_visualization.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/9.3.normal_visualization.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.normal_visualization.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/9.3.normal_visualization.gs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.normal_visualization.gs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/nanosuit
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(instanced_asteroids_ogl demos/instanced_asteroids.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(instanced_asteroids_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(instanced_asteroids_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(instanced_asteroids_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(instanced_asteroids_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(instanced_asteroids_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(instanced_asteroids_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/10.3.asteroids.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.asteroids.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/10.3.asteroids.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.asteroids.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/10.3.planet.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.planet.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/10.3.planet.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.planet.fs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/rock
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/planet
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
endif()