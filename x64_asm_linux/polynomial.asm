    section .note.GNU-stack noalloc noexec nowrite progbits

    segment .data
    scanf_out dd 0
    scanf_format_int db "%d", 0
    scanf_format_float db "%f", 0
    degree_prompt db "Input an integer of polynomial degree:", 0x0a, 0
    value_prompt db "Input a float value:", 0x0a, 0
    coef_prompt db "Input coefficients from high to low (separated by newline):", 0x0a, 0
    result_format db "result = %f", 0x0a, 0

    segment .text
    global main
    global polynomial
    extern printf
    extern scanf
    extern malloc
    extern free

main:
    .idx equ 0
    .degree equ 4
    .value equ 8
    .array equ 12
    push rbp
    mov rbp, rsp
    sub rsp, 32 ; align for 20 bytes
    ; print degree prompt
    lea rdi, [degree_prompt]
    xor rax, rax
    call printf
    ; get degree
    lea rdi, [scanf_format_int] 
    lea rsi, [scanf_out] 
    xor rax, rax
    call scanf
    mov eax, [scanf_out]
    mov [rsp+.degree], eax
    mov [rsp+.idx], eax
    ; print value prompt
    lea rdi, [value_prompt]
    xor rax, rax
    call printf
    ; get value
    lea rdi, [scanf_format_float] 
    lea rsi, [scanf_out] 
    xor rax, rax
    call scanf
    movss xmm0, [scanf_out]
    movss [rsp+.value], xmm0
    ; print coef promp
    lea rdi, [coef_prompt]
    xor rax, rax
    call printf
    ; alloc coef array
    mov eax, [rsp+.degree]
    inc eax
    imul eax, 4
    mov rdi, rax
    call malloc
    mov [rsp+.array], rax
    ; get coefs
.coef_loop:
    lea rdi, [scanf_format_float] 
    lea rsi, [scanf_out] 
    xor rax, rax
    call scanf
    movss xmm0, [scanf_out]
    mov rbx, [rsp+.array]
    mov ecx, [rsp+.idx]
    movss [rbx+rcx*4], xmm0
    dec ecx
    mov [rsp+.idx], ecx
    cmp ecx, 0
    jge .coef_loop
    ; compute polynomial
    mov rdi, [rsp+.array]
    mov esi, [rsp+.degree]
    movss xmm0, [rsp+.value]
    call polynomial
    movss [rsp+.value], xmm0
    mov rdi, [rsp+.array]
    call free
    ; print result
    lea rdi, [result_format]
    movss xmm0, [rsp+.value]
    cvtss2sd xmm0, xmm0
    mov rax, 1
    call printf

    xor rax, rax
    leave
    ret

polynomial: ; (float coef *rdi, float value xmm0, int degree esi)
    movss xmm1, xmm0 ; xmm1 = value
    movss xmm0, [rdi+rsi*4] ; accumulator
    cmp rsi, 0 ; degree = 0 ?
    jz done
more:
    dec esi
    mulss xmm0, xmm1 ; b_k*x
    addss xmm0, [rdi+rsi*4] ; acc + coef_k
    jnz more
done:
    ret