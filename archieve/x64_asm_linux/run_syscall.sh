yasm -f elf64 -g dwarf2 system_call.asm
gcc system_call.o -o system_call -no-pie
rm system_call.o
./system_call
rm system_call