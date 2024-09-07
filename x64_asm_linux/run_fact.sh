yasm -f elf64 -g dwarf2 fact.asm
gcc fact.o -o fact -no-pie
rm fact.o
./fact
rm fact