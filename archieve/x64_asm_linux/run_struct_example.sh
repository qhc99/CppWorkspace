yasm -f elf64 -g dwarf2 struct_example.asm
gcc struct_example.o -o struct_example -no-pie
rm struct_example.o
./struct_example
rm struct_example