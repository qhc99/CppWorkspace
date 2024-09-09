yasm -f elf64 -g dwarf2 array_example.asm
gcc array_example.o -o array_example -no-pie
rm array_example.o
./array_example 5
rm array_example