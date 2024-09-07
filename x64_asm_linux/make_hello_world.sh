yasm -f elf64 -g dwarf2 -l hello_world.lst hello_world.asm
gcc hello_world.o -o hello_world -no-pie