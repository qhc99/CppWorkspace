section .note.GNU-stack noalloc noexec nowrite progbits
    segment .data
x dq 0
scanf_format db "%ld", 0
printf_format db "fact(%ld) = %ld", 0x0a, 0
scanf_hint db "Input a number:", 0x0a, 0

    segment .text
    global main
    global fact
    extern scanf
    extern printf

main:
    push rbp
    mov rbp, rsp
    lea rdi, [scanf_hint]  
    xor eax, eax            
    call printf            
    lea rdi, [scanf_format] ; scanf arg1
    lea rsi, [x] ; scanf arg2
    xor eax, eax
    call scanf
    mov rdi, [x]
    call fact
    lea rdi, [printf_format] ; printf args
    mov rsi, [x]
    mov rdx, rax
    xor eax, eax
    call printf
    xor eax, eax ; set return value
    leave
    ret

fact:
    n equ 8
    push rbp
    mov rbp, rsp
    sub rsp, 16 ; storage for n
    cmp rdi, 1
    jg greater
    mov eax, 1 ; return value 1
    leave
    ret

greater:
    mov [rsp+n], rdi ; save n
    dec rdi
    call fact
    mov rdi, [rsp+n] ; restore
    imul rax, rdi ; fact(n-1)*n
    leave
    ret
