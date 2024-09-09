    section .note.GNU-stack noalloc noexec nowrite progbits
    
    segment .data
    x dq 0
    scanf_format db "%ld", 0
    printf_format db "fact(%ld) = %ld", 0x0a, 0
    scanf_hint db "Input a number:", 0x0a, 0

    segment .text
    global main