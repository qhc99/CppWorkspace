yasm -f elf64 -g dwarf2 float_example.asm
gcc float_example.o -o float_example -no-pie
rm float_example.o
./float_example
rm float_example