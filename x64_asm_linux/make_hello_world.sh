yasm -f elf64 -g dwarf2 -l hello_world.lst hello_world.asm
ld -o hello_world hello_world.o