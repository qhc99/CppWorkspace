yasm -f elf64 -g dwarf2 factorial.asm
gcc factorial.o -o factorial -no-pie
rm factorial.o
./factorial
rm factorial