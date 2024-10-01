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

    add_executable(model_import_ogl demos/model_import.cpp includes/shader_m.h includes/camera.h includes/model.h)
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
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/demos/6.multiple_lights.vs
        ${CMAKE_CURRENT_BINARY_DIR}/6.multiple_lights.vs
        COPYONLY)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demos/nanosuit
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

    # target_link_libraries(main PRIVATE assimp::assimp)
endif()