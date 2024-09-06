    section .data
msg: db "Hello World!", 0x0a, 0

    section .text
    global _start
    extern printf

main:
    push rbp
    mov rbp, rsp
    lea rdi, [msg] ; param 1 of printf
    xor eax, eax ; 0 float param
    call printf
    xor eax, eax ; return 0
    pop rbp
    ret
