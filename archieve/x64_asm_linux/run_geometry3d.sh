yasm -f elf64 -g dwarf2 geometry3d.asm
gcc geometry3d.o -o geometry3d -no-pie
rm geometry3d.o
./geometry3d
rm geometry3d