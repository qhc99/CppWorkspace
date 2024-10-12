segment .data
a dq 151
b dq 310
sum dq 0

segment .text
global _start
_start:
push rbp
mov rbp, rsp
sub rsp, 16
mov rax, 9
add [a], rax
mov rax, [b]
add rax, 10
; desitination: register, memory. source: register, memory, instant number. 
; desitination and source cannot be both memory
add rax, [a] 
mov [sum], rax
mov rax, 0
leave
ret