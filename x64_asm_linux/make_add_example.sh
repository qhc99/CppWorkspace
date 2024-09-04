yasm -f elf64 -g dwarf2 -l add_example.lst add_example.asm
ld -o add_example add_example.o