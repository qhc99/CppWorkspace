    section .note.GNU-stack noalloc noexec nowrite progbits

    segment .data
    scanf_out dd 0
    p1 dd 0.0, 0.0, 0.0
    p2 dd 0.0, 0.0, 0.0
    point_format db "(%f,%f,%f)", 0x0a, 0
    point_prompt db "Input 6 floats separated by newline:", 0x0a, 0
    scanf_format db "%f", 0
    dist_format db "distance = %f", 0x0a, 0
    dotp_format db "dot product result: %f", 0x0a, 0

    segment .text
    global main
    global distance3d
    global dot_product
    global polynomial
    extern printf
    extern scanf

main:
    .idx equ 0
    push rbp
    mov rbp, rsp
    sub rsp, 16
    ; print prompt
    lea rdi, [point_prompt]
    xor rax, rax
    call printf
    ; fill p1
    xor rax, rax
    mov [rsp+.idx], eax ; idx
.fill_p1:
    lea rdi, [scanf_format] 
    lea rsi, [scanf_out] ;
    xor rax, rax
    call scanf
    movss xmm0, [scanf_out]
    mov r8d, [rsp+.idx]
    movss [p1+4*r8d], xmm0
    inc r8d
    mov [rsp+.idx], r8d
    cmp r8d, 3
    jl .fill_p1
    ;fill p2
    xor rax, rax
    mov [rsp+.idx], eax ; idx
.fill_p2:
    lea rdi, [scanf_format] 
    lea rsi, [scanf_out] ;
    xor rax, rax
    call scanf
    movss xmm0, [scanf_out]
    mov r8d, [rsp+.idx]
    movss [p2+4*r8d], xmm0
    inc r8d
    mov [rsp+.idx], r8d
    cmp r8d, 3
    jl .fill_p2
    ; print p1
    lea rdi, [point_format]
    movss xmm0, [p1]
    cvtss2sd xmm0, xmm0
    movss xmm1, [p1+4]
    cvtss2sd xmm1, xmm1
    movss xmm2, [p1+8]
    cvtss2sd xmm2, xmm2
    mov rax, 3  ; Number of vector registers used
    call printf
    ; print p2
    lea rdi, [point_format]
    movss xmm0, [p2]
    cvtss2sd xmm0, xmm0
    movss xmm1, [p2+4]
    cvtss2sd xmm1, xmm1
    movss xmm2, [p2+8]
    cvtss2sd xmm2, xmm2
    mov rax, 3
    call printf
    ; print distance
    lea rdi, [p1]
    lea rsi, [p2]
    call distance3d
    cvtss2sd xmm0, xmm0
    lea rdi, [dist_format]
    mov eax, 1
    call printf
    ; print dot prodcut
    lea rdi, [p1]
    lea rsi, [p2]
    call dot_product
    cvtss2sd xmm0, xmm0
    lea rdi, [dotp_format]
    mov eax, 1
    call printf

    xor rax, rax
    leave
    ret

distance3d: ; (float* rdi, float* rsi)
    movss xmm0, [rdi] ; x from first point
    subss xmm0, [rsi] ; subtract x from second point
    mulss xmm0, xmm0 ; (x1-x2)^2
    movss xmm1, [rdi+4] ; y from first point
    subss xmm1, [rsi+4] ; subtract y from second point
    mulss xmm1, xmm1 ; (y1-y2)^2
    movss xmm2, [rdi+8] ; z from first po int
    subss xmm2, [rsi+8] ; subtract z from second point
    mulss xmm2, xmm2 ; (z1-z2)^2
    addss xmm0, xmm1 ; (x1-x2)^2 + (y1-y2)^2
    addss xmm0, xmm2 ; (x1-x2)^2 + (y1-y2)^2 +  (y1-y2)^2
    sqrtss xmm0, xmm0 ; sqrt sum
    ret

dot_product: ; (float* rdi, float* rsi)
    movss xmm0, [rdi]
    mulss xmm0, [rsi]
    movss xmm1, [rdi+4]
    mulss xmm1, [rsi+4]
    addss xmm0, xmm1
    movss xmm2, [rdi+8]
    mulss xmm2, [rsi+8]
    addss xmm0, xmm2
    ret

polynomial: ; (float coef *rdi, float value xmm0, int degree rsi)
    movsd xmm1, xmm0 ; xmm1 = value
    movsd xmm0, [rdi+rsi*8] ; accumulator
    cmp rsi, 0 ; degree = 0 ?
    jz done
more:
    sub esi, 1
    mulsd xmm0, xmm1 ; b_k*x
    addsd xmm0, [rdi+rsi*8] ; acc + coef_k
    jnz more
done:
    ret