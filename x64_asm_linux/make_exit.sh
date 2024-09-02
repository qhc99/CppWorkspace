yasm -f elf64 -g dwarf2 -l exit.lst exit.asm
ld -o exit exit.o
./exit
echo $?