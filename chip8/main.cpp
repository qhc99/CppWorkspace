#include "chip8.h"
#include "platform.h"
#include <chrono>
#include <iostream>
#include <string>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,concurrency-mt-unsafe)
int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <Scale> <Delay> <ROM>\n";
    std::exit(EXIT_FAILURE);
  }

  int videoScale = std::stoi(argv[1]);
  int cycleDelay = std::stoi(argv[2]);
  char const *romFilename = argv[3];

  Platform platform("CHIP-8 Emulator",
                    static_cast<int>(Chip8::VIDEO_WIDTH * videoScale),
                    static_cast<int>(Chip8::VIDEO_HEIGHT * videoScale),
                    Chip8::VIDEO_WIDTH, Chip8::VIDEO_HEIGHT);

  auto chip8 { std::make_unique<Chip8>() };
  chip8->load_rom(romFilename);

  int videoPitch = sizeof(chip8->video_ref()[0]) * Chip8::VIDEO_WIDTH;

  auto lastCycleTime = std::chrono::high_resolution_clock::now();
  bool quit = false;

  while (!quit) {
    quit = Platform::ProcessInput(chip8->keypad_ref().data());

    auto currentTime = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float, std::chrono::milliseconds::period>(
                   currentTime - lastCycleTime)
                   .count();

    if (dt > static_cast<float>(cycleDelay)) {
      lastCycleTime = currentTime;

      chip8->cycle();

      platform.Update(chip8->video_ref().data(), videoPitch);
    }
  }

  return 0;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,concurrency-mt-unsafe)