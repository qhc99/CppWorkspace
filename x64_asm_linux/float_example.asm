    section .note.GNU-stack noalloc noexec nowrite progbits

    segment .data
    point_format db "%f,%f,f%", 0x0a, 0
    point_prompt db "Input 6 floats (two 3d points)", 0x0a, 0
    dist_format db "distance = %f", 0x0a, 0
    dotp_format db "dot product result: %f", 0x0a, 0
    
    segment .text
    global main

distance3d: ; (double* rdi, double* rsi)
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

dot_product: ; (double* rdi, double* rsi)
    movss xmm0, [rdi]
    mulss xmm0, [rsi]
    movss xmm1, [rdi+4]
    mulss xmm1, [rsi+4]
    addss xmm0, xmm1
    movss xmm2, [rdi+8]
    mulss xmm2, [rsi+8]
    addss xmm0, xmm2
    ret

horner: ; (double coef *rdi, double value xmm0, int degree rsi)
    movs xmm1, xmm0 ; xmm1 = value
    movsv xmm0, [rdi+rsi*8] ; accumulator
    cmp rsi, 0 ; degree = 0 ?
    jz done
more:
    sub esi, 1
    mulsd xmm0, xmm1 ; b_k*x
    addsd xmm0, [rdi+rsi*8] ; acc + coef_k
    jnz more
done:
    ret