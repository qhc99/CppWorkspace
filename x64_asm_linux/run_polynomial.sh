yasm -f elf64 -g dwarf2 polynomial.asm
gcc polynomial.o -o polynomial -no-pie
rm polynomial.o
gdb ./polynomial
rm polynomial