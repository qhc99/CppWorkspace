yasm -f elf64 -g dwarf2 -l mem_example.lst mem_example.asm
ld -o mem_example mem_example.o