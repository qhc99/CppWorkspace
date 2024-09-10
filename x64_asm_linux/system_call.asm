; 32 bit system call: int 0x80
; args: ebx, ecx, edx, esi, edi, ebp
; return: eax

; 64 bit system call
; args: rdi, rsi, rdx, r10, r8, r9
; r10 replace rcx compared with linux ABI
; return: rax
    section .note.GNU-stack noalloc noexec nowrite progbits

    segment .data

hello32: db "Hello world 32 bit", 0x0a
hello64: db "Hello world 64 bit", 0x0a
    segment .text
    global main

main:
    push rbp
    mov rbp, rsp

    mov eax, 4 ; syscall write
    mov ebx, 1 ; file descriptor
    lea ecx, [hello32]
    mov rdx, 19 ; write 19 bytes
    int 0x80

    mov eax, 1 ; syscall 1 is write
    mov edi, 1 ; file descriptor
    lea rsi, [hello64]
    mov edx, 19
    syscall

    mov eax, 60 ; exit
    xor edi, edi ; exit(0)
    syscall

    xor eax, eax 
    leave
    ret