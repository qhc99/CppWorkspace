yasm -f elf64 -g dwarf2 hello_world.asm
gcc hello_world.o -o hello_world -no-pie
rm hello_world.o
./hello_world
rm hello_world