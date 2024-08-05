cmake_minimum_required(VERSION 3.28)

project(chip8 LANGUAGES CXX)

find_package(SDL2 2.30 CONFIG REQUIRED)

add_executable(chip8 main.cpp chip8.h platform.h)
set_target_properties(chip8 PROPERTIES FOLDER "chip8")
target_link_libraries(chip8
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Tetris.ch8
    ${CMAKE_CURRENT_BINARY_DIR}/Tetris.ch8
    COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_opcode.ch8
    ${CMAKE_CURRENT_BINARY_DIR}/test_opcode.ch8
    COPYONLY)