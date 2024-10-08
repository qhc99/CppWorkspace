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

    add_executable(camera_texture_box_ogl demos/camera_texture_box.cpp includes/shader_m.h includes/camera.h)
    set_target_properties(camera_texture_box_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(camera_texture_box_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(camera_texture_box_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(camera_texture_box_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(camera_texture_box_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(camera_texture_box_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/wall.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/wall.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/7.4.camera.vs
        ${CMAKE_CURRENT_BINARY_DIR}/7.4.camera.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/7.4.camera.fs
        ${CMAKE_CURRENT_BINARY_DIR}/7.4.camera.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/awesomeface.png
        ${CMAKE_CURRENT_BINARY_DIR}/awesomeface.png
        COPYONLY)

    add_executable(phong_ogl demos/phong.cpp includes/shader_m.h includes/camera.h)
    set_target_properties(phong_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(phong_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(phong_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(phong_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(phong_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(phong_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.light_cube.vs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.light_cube.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.light_cube.fs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.light_cube.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.lighting_maps.vs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.lighting_maps.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/4.2.lighting_maps.fs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.lighting_maps.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/container2_specular.png
        ${CMAKE_CURRENT_BINARY_DIR}/container2_specular.png
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/container2.png
        ${CMAKE_CURRENT_BINARY_DIR}/container2.png
        COPYONLY)

    add_executable(light_casters_ogl demos/light_casters.cpp includes/shader_m.h includes/camera.h)
    set_target_properties(light_casters_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(light_casters_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(light_casters_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(light_casters_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(light_casters_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(light_casters_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
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

    add_executable(model_import_ogl demos/model_import.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(model_import_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(model_import_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(model_import_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(model_import_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(model_import_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(model_import_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/1.model_loading.vs
        ${CMAKE_CURRENT_BINARY_DIR}/1.model_loading.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/1.model_loading.fs
        ${CMAKE_CURRENT_BINARY_DIR}/1.model_loading.fs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/nanosuit
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(stencil_ogl demos/stencil.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(stencil_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(stencil_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(stencil_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(stencil_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(stencil_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(stencil_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
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

    add_executable(framebuffer_ogl demos/framebuffer.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(framebuffer_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(framebuffer_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(framebuffer_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(framebuffer_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(framebuffer_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(framebuffer_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
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
endif()