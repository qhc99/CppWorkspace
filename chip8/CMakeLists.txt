cmake_minimum_required(VERSION 3.28)

project(chip8 LANGUAGES CXX)

# [cmake] sdl2 provides CMake targets:
# [cmake] 
# [cmake]     find_package(SDL2 CONFIG REQUIRED)
# [cmake]     target_link_libraries(main
# [cmake]         PRIVATE
# [cmake]         $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
# [cmake]         $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
# [cmake]     )