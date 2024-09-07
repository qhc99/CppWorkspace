yasm -f elf64 -g dwarf2 -l fact.lst fact.asm
gcc fact.o -o fact -no-pie