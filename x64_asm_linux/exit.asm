;Program : exit
;Exe cutes the exit system call
;No input
;Output : only the exit status ($? in the she l l )
segment .text
global _start
_start :
mov eax , 1 ; 1 is the exit sys call number
mov ebx , 4 ; the status value to return
int 0x80 ; execute a system call

