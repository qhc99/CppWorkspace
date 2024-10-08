cmake_minimum_required(VERSION 3.28)

if(CMAKE_SYSTEM_NAME MATCHES Windows)
    find_package(SDL2 2.30 CONFIG REQUIRED)

    project(chip8 LANGUAGES CXX)

    add_executable(chip8 main.cpp chip8.h platform.h)
    set_target_properties(chip8 PROPERTIES FOLDER "chip8")
    target_compile_options(chip8 PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_link_options(chip8
        PRIVATE
        ${COMMON_LINK_OPTIONS}
        $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Windows>>:-Xlinker;/subsystem:windows>
    )
    target_link_libraries(chip8
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        workspace_pch
    )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Tetris.ch8
        ${CMAKE_CURRENT_BINARY_DIR}/Tetris.ch8
        COPYONLY)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_opcode.ch8
        ${CMAKE_CURRENT_BINARY_DIR}/test_opcode.ch8
        COPYONLY)

    message(STATUS ">>> Added chip8")
endif()