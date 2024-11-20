cmake_minimum_required(VERSION 3.28)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    project(opengl_learn LANGUAGES CXX)

    find_package(OpenGL REQUIRED COMPONENTS OpenGL)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(Stb REQUIRED)
    find_package(glm CONFIG REQUIRED)
    find_package(assimp CONFIG REQUIRED)
    find_package(Freetype REQUIRED)

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
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/models/nanosuit
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
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/models/rock
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/models/planet
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(msaa_framebuffer_ogl demos/msaa_framebuffer.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(msaa_framebuffer_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(msaa_framebuffer_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
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
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/models/nanosuit
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(ssao_ogl demos/screen_space_ambient_occlusion.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(ssao_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(ssao_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(ssao_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(ssao_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(ssao_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(ssao_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.ssao_geometry.vs
        ${CMAKE_CURRENT_BINARY_DIR}/9.ssao_geometry.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.ssao_geometry.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.ssao_geometry.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.ssao.vs
        ${CMAKE_CURRENT_BINARY_DIR}/9.ssao.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.ssao.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.ssao.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.ssao_lighting.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.ssao_lighting.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/9.ssao_blur.fs
        ${CMAKE_CURRENT_BINARY_DIR}/9.ssao_blur.fs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/models/backpack
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(pbr_disney_ogl demos/PBR_disney.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(pbr_disney_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(pbr_disney_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(pbr_disney_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(pbr_disney_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(pbr_disney_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(pbr_disney_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/1.1.pbr.vs
        ${CMAKE_CURRENT_BINARY_DIR}/1.1.pbr.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/1.1.pbr.fs
        ${CMAKE_CURRENT_BINARY_DIR}/1.1.pbr.fs
        COPYONLY)

    add_executable(image_based_lighting_ogl demos/image_based_lighting.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(image_based_lighting_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(image_based_lighting_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(image_based_lighting_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(image_based_lighting_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(image_based_lighting_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(image_based_lighting_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.pbr.vs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.pbr.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.pbr.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.pbr.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.cubemap.vs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.cubemap.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.equirectangular_to_cubemap.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.equirectangular_to_cubemap.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.irradiance_convolution.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.irradiance_convolution.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.prefilter.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.prefilter.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.brdf.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.brdf.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.brdf.vs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.brdf.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.background.fs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.background.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/2.2.1.background.vs
        ${CMAKE_CURRENT_BINARY_DIR}/2.2.1.background.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/newport_loft.hdr
        ${CMAKE_CURRENT_BINARY_DIR}/newport_loft.hdr
        COPYONLY)

    add_executable(error_handling_ogl demos/error_handling.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(error_handling_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(error_handling_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(error_handling_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(error_handling_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(error_handling_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(error_handling_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/wood.png
        ${CMAKE_CURRENT_BINARY_DIR}/wood.png
        COPYONLY)

    add_executable(font_ttf_ogl demos/font_ttf.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h)
    set_target_properties(font_ttf_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(font_ttf_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only Freetype::Freetype)
    target_include_directories(font_ttf_ogl PRIVATE includes)
    target_compile_options(font_ttf_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(font_ttf_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/Antonio-Bold.ttf
        ${CMAKE_CURRENT_BINARY_DIR}/Antonio-Bold.ttf
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/text.vs
        ${CMAKE_CURRENT_BINARY_DIR}/text.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/text.fs
        ${CMAKE_CURRENT_BINARY_DIR}/text.fs
        COPYONLY)

    add_executable(pbr_area_light_ogl demos/pbr_area_light.cpp includes/shader_m.h includes/camera.h includes/model.h includes/shader.h includes/colors.hpp includes/ltc_matrix.hpp)
    set_target_properties(pbr_area_light_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(pbr_area_light_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(pbr_area_light_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(pbr_area_light_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(pbr_area_light_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(pbr_area_light_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/concreteTexture.png
        ${CMAKE_CURRENT_BINARY_DIR}/concreteTexture.png
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.multi_area_light.vs
        ${CMAKE_CURRENT_BINARY_DIR}/7.multi_area_light.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.multi_area_light.fs
        ${CMAKE_CURRENT_BINARY_DIR}/7.multi_area_light.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.light_plane.vs
        ${CMAKE_CURRENT_BINARY_DIR}/7.light_plane.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/7.light_plane.fs
        ${CMAKE_CURRENT_BINARY_DIR}/7.light_plane.fs
        COPYONLY)

    add_executable(animation_ogl demos/animation.cpp)
    set_target_properties(animation_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(animation_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only assimp::assimp)
    target_include_directories(animation_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(animation_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(animation_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(animation_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/anim_model.vs
        ${CMAKE_CURRENT_BINARY_DIR}/anim_model.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/anim_model.fs
        ${CMAKE_CURRENT_BINARY_DIR}/anim_model.fs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/models/vampire
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    add_executable(weighted_OIT_ogl demos/weighted_OIT.cpp)
    set_target_properties(weighted_OIT_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(weighted_OIT_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(weighted_OIT_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(weighted_OIT_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(weighted_OIT_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(weighted_OIT_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/solid.vs
        ${CMAKE_CURRENT_BINARY_DIR}/solid.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/solid.fs
        ${CMAKE_CURRENT_BINARY_DIR}/solid.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/transparent.vs
        ${CMAKE_CURRENT_BINARY_DIR}/transparent.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/transparent.fs
        ${CMAKE_CURRENT_BINARY_DIR}/transparent.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/composite.vs
        ${CMAKE_CURRENT_BINARY_DIR}/composite.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/composite.fs
        ${CMAKE_CURRENT_BINARY_DIR}/composite.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/screen.vs
        ${CMAKE_CURRENT_BINARY_DIR}/screen.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/screen.fs
        ${CMAKE_CURRENT_BINARY_DIR}/screen.fs
        COPYONLY)

    add_executable(cascaded_shadow_mapping_ogl demos/cascaded_shadow_mapping.cpp)
    set_target_properties(cascaded_shadow_mapping_ogl PROPERTIES FOLDER "opengl")
    target_link_libraries(cascaded_shadow_mapping_ogl PRIVATE OpenGL::GL glad::glad glfw glm::glm-header-only)
    target_include_directories(cascaded_shadow_mapping_ogl PRIVATE ${Stb_INCLUDE_DIR} includes)
    target_compile_options(cascaded_shadow_mapping_ogl PRIVATE ${COMMON_COMPILE_OPTIONS} ${SUPPRESS_COMPILE_WARNINGS_OPTION} $<$<CXX_COMPILER_ID:MSVC>:/WX->)
    target_link_options(cascaded_shadow_mapping_ogl PRIVATE ${COMMON_LINK_OPTIONS})
    target_compile_definitions(cascaded_shadow_mapping_ogl PRIVATE STB_IMAGE_IMPLEMENTATION)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.shadow_mapping.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.shadow_mapping.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.shadow_mapping.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.shadow_mapping.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.shadow_mapping_depth.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.shadow_mapping_depth.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.shadow_mapping_depth.gs
        ${CMAKE_CURRENT_BINARY_DIR}/10.shadow_mapping_depth.gs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.shadow_mapping_depth.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.shadow_mapping_depth.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.debug_quad.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.debug_quad.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.debug_quad_depth.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.debug_quad_depth.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.debug_cascade.vs
        ${CMAKE_CURRENT_BINARY_DIR}/10.debug_cascade.vs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/shaders/10.debug_cascade.fs
        ${CMAKE_CURRENT_BINARY_DIR}/10.debug_cascade.fs
        COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/textures/wood.png
        ${CMAKE_CURRENT_BINARY_DIR}/wood.png
        COPYONLY)
endif()