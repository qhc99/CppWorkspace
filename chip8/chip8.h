#ifndef CPP_WORKSPACE_CHIP8
#define CPP_WORKSPACE_CHIP8

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <cstring>

class Chip8 {
  static constexpr unsigned int START_ADDRESS = 0x200;
  static constexpr unsigned int FONTSET_SIZE = 80;
  static constexpr unsigned int FONTSET_START_ADDRESS = 0x50;

  static constexpr std::array<uint8_t, FONTSET_SIZE> fontset = {{
      0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
      0x20, 0x60, 0x20, 0x20, 0x70, // 1
      0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
      0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
      0x90, 0x90, 0xF0, 0x10, 0x10, // 4
      0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
      0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
      0xF0, 0x10, 0x20, 0x40, 0x40, // 7
      0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
      0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
      0xF0, 0x90, 0xF0, 0x90, 0x90, // A
      0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
      0xF0, 0x80, 0x80, 0x80, 0xF0, // C
      0xE0, 0x90, 0x90, 0x90, 0xE0, // D
      0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
      0xF0, 0x80, 0xF0, 0x80, 0x80  // F
  }};

  using Chip8Func = void (Chip8::*)();
  std::array<Chip8Func, 0xF + 1> table{};
  std::array<Chip8Func, 0xE + 1> table0{};
  std::array<Chip8Func, 0xE + 1> table8{};
  std::array<Chip8Func, 0xE + 1> tableE{};
  std::array<Chip8Func, 0x65 + 1> tableF{};

  std::array<uint8_t, 16> registers{};
  std::array<uint8_t, 40960> memory{};
  uint16_t index{};
  uint16_t pc;
  std::array<uint16_t, 16> stack{};
  uint8_t sp{};
  uint8_t delayTimer{};
  uint8_t soundTimer{};
  std::array<uint8_t, 16> keypad{};
  std::array<uint32_t, static_cast<size_t>(64 * 32)> video{};
  uint16_t opcode{};
  std::default_random_engine randGen;
  std::uniform_int_distribution<uint16_t> randByte;

public:
  static constexpr unsigned int VIDEO_WIDTH = 64;
  static constexpr unsigned int VIDEO_HEIGHT = 32;

  std::array<uint32_t, static_cast<size_t>(64 * 32)> &video_ref() {
    return video;
  }

  std::array<uint8_t, 16> &keypad_ref() { return keypad; }

  Chip8()
      : pc(START_ADDRESS),
        randGen(std::chrono::system_clock::now().time_since_epoch().count()),
        randByte(std::uniform_int_distribution<uint16_t>(0, 255U)) {
    // Load fonts into memory
    for (unsigned int i = 0; i < FONTSET_SIZE; ++i) {
      memory.at(FONTSET_START_ADDRESS + i) = fontset.at(i);
    }

    // Set up function pointer table
    table.at(0x0) = &Chip8::Table0;
    table[0x1] = &Chip8::OP_1nnn;
    table[0x2] = &Chip8::OP_2nnn;
    table[0x3] = &Chip8::OP_3xkk;
    table[0x4] = &Chip8::OP_4xkk;
    table[0x5] = &Chip8::OP_5xy0;
    table[0x6] = &Chip8::OP_6xkk;
    table[0x7] = &Chip8::OP_7xkk;
    table[0x8] = &Chip8::Table8;
    table[0x9] = &Chip8::OP_9xy0;
    table[0xA] = &Chip8::OP_Annn;
    table[0xB] = &Chip8::OP_Bnnn;
    table[0xC] = &Chip8::OP_Cxkk;
    table[0xD] = &Chip8::OP_Dxyn;
    table[0xE] = &Chip8::TableE;
    table[0xF] = &Chip8::TableF;

    for (size_t i = 0; i <= 0xE; i++) {
      table0.at(i) = &Chip8::OP_NULL;
      table8.at(i) = &Chip8::OP_NULL;
      tableE.at(i) = &Chip8::OP_NULL;
    }

    table0[0x0] = &Chip8::OP_00E0;
    table0[0xE] = &Chip8::OP_00EE;

    table8[0x0] = &Chip8::OP_8xy0;
    table8[0x1] = &Chip8::OP_8xy1;
    table8[0x2] = &Chip8::OP_8xy2;
    table8[0x3] = &Chip8::OP_8xy3;
    table8[0x4] = &Chip8::OP_8xy4;
    table8[0x5] = &Chip8::OP_8xy5;
    table8[0x6] = &Chip8::OP_8xy6;
    table8[0x7] = &Chip8::OP_8xy7;
    table8[0xE] = &Chip8::OP_8xyE;

    tableE[0x1] = &Chip8::OP_ExA1;
    tableE[0xE] = &Chip8::OP_Ex9E;

    for (size_t i = 0; i <= 0x65; i++) {
      tableF.at(i) = &Chip8::OP_NULL;
    }

    tableF[0x07] = &Chip8::OP_Fx07;
    tableF[0x0A] = &Chip8::OP_Fx0A;
    tableF[0x15] = &Chip8::OP_Fx15;
    tableF[0x18] = &Chip8::OP_Fx18;
    tableF[0x1E] = &Chip8::OP_Fx1E;
    tableF[0x29] = &Chip8::OP_Fx29;
    tableF[0x33] = &Chip8::OP_Fx33;
    tableF[0x55] = &Chip8::OP_Fx55;
    tableF[0x65] = &Chip8::OP_Fx65;
  }

  void Table0() { ((*this).*(table0.at(opcode & 0x000FU)))(); }

  void Table8() { ((*this).*(table8.at(opcode & 0x000FU)))(); }

  void TableE() { ((*this).*(tableE.at(opcode & 0x000FU)))(); }

  void TableF() { ((*this).*(tableF.at(opcode & 0x00FFU)))(); }

  void OP_NULL() {}

  void OP_00E0() { std::memset(video.data(), 0, sizeof(video)); }

  void OP_00EE() {
    --sp;
    pc = stack.at(sp);
  }

  void OP_1nnn() {
    uint16_t address = opcode & 0x0FFFU;
    pc = address;
  }

  void OP_2nnn() {
    uint16_t address = opcode & 0x0FFFU;
    stack.at(sp) = pc;
    ++sp;
    pc = address;
  }

  void OP_3xkk() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t byte = opcode & 0x00FFU;
    if (registers.at(Vx) == byte) {
      pc += 2;
    }
  }

  void OP_4xkk() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t byte = opcode & 0x00FFU;
    if (registers.at(Vx) != byte) {
      pc += 2;
    }
  }

  void OP_5xy0() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;

    if (registers.at(Vx) == registers.at(Vy)) {
      pc += 2;
    }
  }

  void OP_6xkk() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t byte = opcode & 0x00FFU;
    registers.at(Vx) = byte;
  }

  void OP_7xkk() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t byte = opcode & 0x00FFU;
    registers.at(Vx) += byte;
  }

  void OP_8xy0() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;
    registers.at(Vx) = registers.at(Vy);
  }

  void OP_8xy1() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;
    registers.at(Vx) |= registers.at(Vy);
  }

  void OP_8xy2() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;
    registers.at(Vx) &= registers.at(Vy);
  }

  void OP_8xy3() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;
    registers.at(Vx) ^= registers.at(Vy);
  }

  void OP_8xy4() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;

    uint16_t sum = registers.at(Vx) + registers.at(Vy);

    if (sum > 255U) {
      registers.at(0xF) = 1;
    } else {
      registers.at(0xF) = 0;
    }

    registers.at(Vx) = sum & 0xFFU;
  }

  void OP_8xy5() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;

    if (registers.at(Vx) > registers.at(Vy)) {
      registers.at(0xF) = 1;
    } else {
      registers.at(0xF) = 0;
    }

    registers.at(Vx) -= registers.at(Vy);
  }

  void OP_8xy6() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    // Save LSB in VF
    registers.at(0xF) = (registers.at(Vx) & 0x1U);

    registers.at(Vx) >>= 1U;
  }

  void OP_8xy7() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;

    if (registers.at(Vy) > registers.at(Vx)) {
      registers.at(0xF) = 1;
    } else {
      registers.at(0xF) = 0;
    }

    registers.at(Vx) = registers.at(Vy) - registers.at(Vx);
  }

  void OP_8xyE() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    // Save MSB in VF
    registers.at(0xF) = (registers.at(Vx) & 0x80U) >> 7U;
    registers.at(Vx) <<= 1U;
  }

  void OP_9xy0() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;

    if (registers.at(Vx) != registers.at(Vy)) {
      pc += 2;
    }
  }

  void OP_Annn() {
    uint16_t address = opcode & 0x0FFFU;
    index = address;
  }

  void OP_Bnnn() {
    uint16_t address = opcode & 0x0FFFU;
    pc = registers.at(0) + address;
  }

  void OP_Cxkk() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t byte = opcode & 0x00FFU;
    registers.at(Vx) = static_cast<uint8_t>(randByte(randGen)) & byte;
  }

  void OP_Dxyn() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t Vy = (opcode & 0x00F0U) >> 4U;
    uint8_t height = opcode & 0x000FU;
    // Wrap if going beyond screen boundaries
    uint8_t xPos = registers.at(Vx) % VIDEO_WIDTH;
    uint8_t yPos = registers.at(Vy) % VIDEO_HEIGHT;

    registers[0xF] = 0;

    for (unsigned int row = 0; row < height; ++row) {
      uint8_t spriteByte = memory.at(index + row);

      for (unsigned int col = 0; col < 8; ++col) {
        uint8_t spritePixel = spriteByte & (0x80U >> col);
        // Fix bug of reference code
        if((yPos + row) * VIDEO_WIDTH + (xPos + col) >= video.size()){
          continue;
        }
        uint32_t *screenPixel =
            &video.at((yPos + row) * VIDEO_WIDTH + (xPos + col));
        // Sprite pixel is on
        if (spritePixel != 0U) {
          // Screen pixel also on - collision
          if (*screenPixel == 0xFFFFFFFF) {
            registers[0xF] = 1;
          }
          // Effectively XOR with the sprite pixel
          *screenPixel ^= 0xFFFFFFFF;
        }
      }
    }
  }

  void OP_Ex9E() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t key = registers.at(Vx);
    if (keypad.at(key) != 0U) {
      pc += 2;
    }
  }

  void OP_ExA1() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t key = registers.at(Vx);
    if (keypad.at(key) == 0U) {
      pc += 2;
    }
  }

  void OP_Fx07() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    registers.at(Vx) = delayTimer;
  }

  void OP_Fx0A() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;

    if (keypad.at(0) != 0U) {
      registers.at(Vx) = 0;
    } else if (keypad.at(1) != 0U) {
      registers.at(Vx) = 1;
    } else if (keypad.at(2) != 0U) {
      registers.at(Vx) = 2;
    } else if (keypad.at(3) != 0U) {
      registers.at(Vx) = 3;
    } else if (keypad.at(4) != 0U) {
      registers.at(Vx) = 4;
    } else if (keypad.at(5) != 0U) {
      registers.at(Vx) = 5;
    } else if (keypad.at(6) != 0U) {
      registers.at(Vx) = 6;
    } else if (keypad.at(7) != 0U) {
      registers.at(Vx) = 7;
    } else if (keypad.at(8) != 0U) {
      registers.at(Vx) = 8;
    } else if (keypad.at(9) != 0U) {
      registers.at(Vx) = 9;
    } else if (keypad.at(10) != 0U) {
      registers.at(Vx) = 10;
    } else if (keypad.at(11) != 0U) {
      registers.at(Vx) = 11;
    } else if (keypad.at(12) != 0U) {
      registers.at(Vx) = 12;
    } else if (keypad.at(13) != 0U) {
      registers.at(Vx) = 13;
    } else if (keypad.at(14) != 0U) {
      registers.at(Vx) = 14;
    } else if (keypad.at(15) != 0U) {
      registers.at(Vx) = 15;
    } else {
      pc -= 2;
    }
  }

  void OP_Fx15() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    delayTimer = registers.at(Vx);
  }

  void OP_Fx18() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    soundTimer = registers.at(Vx);
  }

  void OP_Fx1E() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    index += registers.at(Vx);
  }

  void OP_Fx29() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t digit = registers.at(Vx);

    index = FONTSET_START_ADDRESS + (5 * digit);
  }

  void OP_Fx33() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;
    uint8_t value = registers.at(Vx);

    // Ones-place
    memory.at(index + 2) = value % 10;
    value /= 10;

    // Tens-place
    memory.at(index + 1) = value % 10;
    value /= 10;

    // Hundreds-place
    memory.at(index) = value % 10;
  }

  void OP_Fx55() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;

    for (uint8_t i = 0; i <= Vx; ++i) {
      memory.at(index + i) = registers.at(i);
    }
  }

  void OP_Fx65() {
    uint8_t Vx = (opcode & 0x0F00U) >> 8U;

    for (uint8_t i = 0; i <= Vx; ++i) {
      registers.at(i) = memory.at(index + i);
    }
  }

  void cycle() {
    // Fetch
    // NOLINTNEXTLINE hicpp-signed-bitwise
    opcode = (memory.at(pc) << 8U) |
             memory.at(pc + 1); 
    // Increment the PC before we execute anything
    pc += 2;
    // Decode and Execute
    ((*this).*(table.at((opcode & 0xF000U) >> 12U)))();
    // Decrement the delay timer if it's been set
    if (delayTimer > 0) {
      --delayTimer;
    }
    // Decrement the sound timer if it's been set
    if (soundTimer > 0) {
      --soundTimer;
    }
  }

  void load_rom(char const *filename) {
    // Open the file as a stream of binary and move the file pointer to the end
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (file.is_open()) {
      // Get size of file and allocate a buffer to hold the contents
      std::streampos size = file.tellg();
      std::vector<char> buffer(size);

      // Go back to the beginning of the file and fill the buffer
      file.seekg(0, std::ios::beg);
      file.read(buffer.data(), size);
      file.close();

      // Load the ROM contents into the Chip8's memory, starting at 0x200
      for (long i = 0; i < size; ++i) {
        memory.at(START_ADDRESS + i) = buffer.at(i);
      }
    }
  }
};

#endif