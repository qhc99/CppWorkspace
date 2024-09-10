yasm -f elf64 -g dwarf2 float_example1.asm
gcc float_example1.o -o float_example1 -no-pie
rm float_example1.o
./float_example1
rm float_example1