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

    add_executable(phong_light_casters_ogl demos/phong_light_casters.cpp includes/shader_m.h includes/camera.h)
    set_target_properties(phong_light_casters_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(phong_light_casters_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(phong_light_casters_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(phong_light_casters_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(phong_light_casters_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(phong_light_casters_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/4.2.light_cube.vs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.light_cube.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/4.2.light_cube.fs
        ${CMAKE_CURRENT_BINARY_DIR}/4.2.light_cube.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/6.multiple_lights.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.multiple_lights.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/6.multiple_lights.fs
        ${CMAKE_CURRENT_BINARY_DIR}/6.multiple_lights.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/container2_specular.png
        ${CMAKE_CURRENT_BINARY_DIR}/container2_specular.png
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/container2.png
        ${CMAKE_CURRENT_BINARY_DIR}/container2.png
        COPYONLY)

    add_executable(stencil_outline_ogl demos/stencil_outline.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(stencil_outline_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(stencil_outline_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(stencil_outline_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(stencil_outline_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(stencil_outline_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(stencil_outline_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.stencil_testing.vs
        ${CMAKE_CURRENT_BINARY_DIR}/2.stencil_testing.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.stencil_testing.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.stencil_testing.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.stencil_single_color.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.stencil_single_color.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/marble.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/marble.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/metal.png
        ${CMAKE_CURRENT_BINARY_DIR}/metal.png
        COPYONLY)

    add_executable(blend_transparent_ogl demos/blend_transparent.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(blend_transparent_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(blend_transparent_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(blend_transparent_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(blend_transparent_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(blend_transparent_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(blend_transparent_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/3.2.blending.vs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.blending.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/3.2.blending.fs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.blending.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/marble.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/marble.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/metal.png
        ${CMAKE_CURRENT_BINARY_DIR}/metal.png
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/window.png
        ${CMAKE_CURRENT_BINARY_DIR}/window.png
        COPYONLY)

    add_executable(framebuffer_rearview_ogl demos/framebuffer_rearview.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(framebuffer_rearview_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(framebuffer_rearview_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(framebuffer_rearview_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(framebuffer_rearview_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(framebuffer_rearview_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(framebuffer_rearview_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/5.2.framebuffers.vs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/5.2.framebuffers.fs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/5.2.framebuffers_screen.vs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers_screen.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/5.2.framebuffers_screen.fs
        ${CMAKE_CURRENT_BINARY_DIR}/5.2.framebuffers_screen.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/container.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/container.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/metal.png
        ${CMAKE_CURRENT_BINARY_DIR}/metal.png
        COPYONLY)

    add_executable(skybox_ogl demos/skybox.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(skybox_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(skybox_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(skybox_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(skybox_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(skybox_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(skybox_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/6.2.cubemaps.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.2.cubemaps.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/6.2.cubemaps.fs
        ${CMAKE_CURRENT_BINARY_DIR}/6.2.cubemaps.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/6.2.skybox.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.2.skybox.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/6.2.skybox.fs
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
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.3.default.vs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.default.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.3.default.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.default.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.3.normal_visualization.vs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.normal_visualization.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.3.normal_visualization.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.3.normal_visualization.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.3.normal_visualization.gs
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
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.3.asteroids.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.asteroids.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.3.asteroids.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.asteroids.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.3.planet.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.planet.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.3.planet.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.3.planet.fs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/rock
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/planet
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(msaa_framebuffer_ogl demos/msaa_framebuffer.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(msaa_framebuffer_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(msaa_framebuffer_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(msaa_framebuffer_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(msaa_framebuffer_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(msaa_framebuffer_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(msaa_framebuffer_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/11.2.anti_aliasing.vs
        ${CMAKE_CURRENT_BINARY_DIR}/11.2.anti_aliasing.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/11.2.anti_aliasing.fs
        ${CMAKE_CURRENT_BINARY_DIR}/11.2.anti_aliasing.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/11.2.aa_post.vs
        ${CMAKE_CURRENT_BINARY_DIR}/11.2.aa_post.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/11.2.aa_post.fs
        ${CMAKE_CURRENT_BINARY_DIR}/11.2.aa_post.fs
        COPYONLY)

    add_executable(shadow_cubmap_ogl demos/shadow_cubmap.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(shadow_cubmap_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(shadow_cubmap_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(shadow_cubmap_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(shadow_cubmap_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(shadow_cubmap_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(shadow_cubmap_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/3.2.2.point_shadows.vs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.2.point_shadows.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/3.2.2.point_shadows.fs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.2.point_shadows.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/3.2.2.point_shadows_depth.vs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.2.point_shadows_depth.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/3.2.2.point_shadows_depth.fs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.2.point_shadows_depth.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/3.2.2.point_shadows_depth.gs
        ${CMAKE_CURRENT_BINARY_DIR}/3.2.2.point_shadows_depth.gs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/wood.png
        ${CMAKE_CURRENT_BINARY_DIR}/wood.png
        COPYONLY)

    add_executable(normal_mapping_ogl demos/normal_mapping.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(normal_mapping_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(normal_mapping_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(normal_mapping_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(normal_mapping_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(normal_mapping_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(normal_mapping_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/4.normal_mapping.vs
        ${CMAKE_CURRENT_BINARY_DIR}/4.normal_mapping.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/4.normal_mapping.fs
        ${CMAKE_CURRENT_BINARY_DIR}/4.normal_mapping.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/brickwall_normal.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/brickwall_normal.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/brickwall.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/brickwall.jpg
        COPYONLY)

    add_executable(parallax_occlusion_mapping_ogl demos/parallax_occlusion_mapping.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(parallax_occlusion_mapping_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(parallax_occlusion_mapping_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(parallax_occlusion_mapping_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(parallax_occlusion_mapping_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(parallax_occlusion_mapping_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(parallax_occlusion_mapping_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/5.3.parallax_mapping.vs
        ${CMAKE_CURRENT_BINARY_DIR}/5.3.parallax_mapping.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/5.3.parallax_mapping.fs
        ${CMAKE_CURRENT_BINARY_DIR}/5.3.parallax_mapping.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/bricks2.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/bricks2.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/bricks2_normal.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/bricks2_normal.jpg
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/bricks2_disp.jpg
        ${CMAKE_CURRENT_BINARY_DIR}/bricks2_disp.jpg
        COPYONLY)

    add_executable(hdr_bloom_ogl demos/hdr_bloom.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(hdr_bloom_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(hdr_bloom_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(hdr_bloom_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(hdr_bloom_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(hdr_bloom_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(hdr_bloom_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.bloom.vs
        ${CMAKE_CURRENT_BINARY_DIR}/7.bloom.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.bloom.fs
        ${CMAKE_CURRENT_BINARY_DIR}/7.bloom.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.light_box.fs
        ${CMAKE_CURRENT_BINARY_DIR}/7.light_box.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.blur.vs
        ${CMAKE_CURRENT_BINARY_DIR}/7.blur.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.blur.fs
        ${CMAKE_CURRENT_BINARY_DIR}/7.blur.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.bloom_final.vs
        ${CMAKE_CURRENT_BINARY_DIR}/7.bloom_final.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.bloom_final.fs
        ${CMAKE_CURRENT_BINARY_DIR}/7.bloom_final.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/wood.png
        ${CMAKE_CURRENT_BINARY_DIR}/wood.png
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/container2.png
        ${CMAKE_CURRENT_BINARY_DIR}/container2.png
        COPYONLY)

    add_executable(deferred_shading_ogl demos/deferred_shading.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(deferred_shading_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(deferred_shading_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(deferred_shading_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(deferred_shading_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(deferred_shading_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(deferred_shading_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/8.2.g_buffer.vs
        ${CMAKE_CURRENT_BINARY_DIR}/8.2.g_buffer.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/8.2.g_buffer.fs
        ${CMAKE_CURRENT_BINARY_DIR}/8.2.g_buffer.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/8.2.deferred_shading.vs
        ${CMAKE_CURRENT_BINARY_DIR}/8.2.deferred_shading.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/8.2.deferred_shading.fs
        ${CMAKE_CURRENT_BINARY_DIR}/8.2.deferred_shading.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/8.2.deferred_light_box.vs
        ${CMAKE_CURRENT_BINARY_DIR}/8.2.deferred_light_box.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/8.2.deferred_light_box.fs
        ${CMAKE_CURRENT_BINARY_DIR}/8.2.deferred_light_box.fs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/nanosuit
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
endif()