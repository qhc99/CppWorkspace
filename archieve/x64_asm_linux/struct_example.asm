    section .note.GNU-stack noalloc noexec nowrite progbits

    segment .data
    name db "Calvin", 0
    address db "12 Mockingbrid Lane", 0
    balance dd 12500
    format db "%s, %s, %d", 0x0a, 0

    struc Customer
    .id resd 1
    .name resb 65
    .addr resb 65
    align 4
    .balance resd 1
    endstruc
    ptr dq 0

    static_customer istruc Customer
    at Customer.id, dd 7
    at Customer.name, db "static customer", 0
    at Customer.addr, db "Homeless", 0
    at Customer.balance, dd 5
    iend

    segment .text
    global main
    extern malloc, strcpy, printf
main:
    push rbp
    mov rbp, rsp

    mov rdi, Customer_size
    call malloc
    mov [ptr], rax

    lea rdi, [rax+Customer.name]
    lea rsi, [name]
    call strcpy
    
    mov rax, [ptr]
    lea rdi, [rax+Customer.addr]
    lea rsi, [address]
    call strcpy
    
    mov rax, [ptr]
    mov edx, [balance]
    mov [rax+Customer.balance], edx

    mov rax, [ptr]
    lea rdi, [format]
    lea rsi, [rax+Customer.name]
    lea rdx, [rax+Customer.addr]
    mov ecx, [rax+Customer.balance]
    xor eax, eax
    call printf

    lea edi, [format]
    lea esi, [static_customer+Customer.name]
    lea rdx, [static_customer+Customer.addr]
    mov rcx, [static_customer+Customer.balance]
    xor eax, eax
    call printf

    xor eax, eax
    leave
    ret
