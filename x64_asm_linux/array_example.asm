; This application allocates an array using malloc, fills
; the array with random numbers by calling random and computes the
; minimum value in the array. If the array size is less than or equal to 20,
; it prints the values in the array.

    section .note.GNU-stack noalloc noexec nowrite progbits

    segment .text
    global create
    global fill
    global print
    global min
    global main
    extern printf
    extern malloc
    extern random
    extern atoi
    extern free

main:
    .array equ 0
    .size equ 8
    push rbp
    mov rbp, rsp
    sub rsp, 16
; set default size
    mov ecx, 10
    mov [rsp+.size], rcx
; check for argv [1] providing a size
    cmp edi, 2
    jl .nosize
    mov rdi, [rsi+8]
    call atoi
    mov [rsp+.size], rax
.nosize:
; create the array
    mov rdi, [rsp+.size]
    call create
    mov [rsp+.array], rax
; fill the array with random numbers
    mov rdi, rax
    mov rsi, [rsp+.size]
    call fill
; if size <= 20 print the array
    mov rsi, [rsp+.size]
    cmp rsi, 20
    jg .toobig
    mov rdi, [rsp+.array]
    call print
.toobig:
; print the minimum
    segment .data
.format:
    db "min:%ld", 0xa, 0
    segment .text
    mov rdi, [rsp+.array]
    mov rsi, [rsp+.size]
    call min
    lea rdi, [.format]
    mov rsi, rax
    call printf
    mov rdi, [rsp+.array]
    call free
    leave
    ret
    

; array = create(size)
create:
    push rbp
    mov rbp, rsp
    imul rdi, 4
    call malloc
    leave
    ret

; fill(array, size)
fill:
    .array equ 0 ; local variables, 24 bytes
    .size equ 8
    .i equ 16
    push rbp
    mov rbp, rsp
    sub rsp, 32 ; local variables + alignment
    mov [rsp+.array], rdi
    mov [rsp+.size], rsi
    xor ecx, ecx
.more:
    mov [rsp+.i], rcx
    call random
    mov rcx, [rsp+.i]
    mov rdi, [rsp+.array]
    mov [rdi+rcx*4], eax
    inc rcx
    cmp rcx, [rsp+.size]
    jl .more
    leave
    ret


; print(array,size)
print:
    .array equ 0
    .size equ 8
    .i equ 16
    push rbp
    mov rbp, rsp
    sub rsp, 32
    mov [rsp+.array], rdi
    mov [rsp+.size], rsi
    xor ecx, ecx
    mov [rsp+ .i] , rcx
    segment .data
.format:
    db "%10d", 0x0a , 0
    segment .text
.more:
    lea rdi, [.format]
    mov rdx, [rsp+.array]
    mov rcx, [rsp+.i]
    mov esi, [rdx+rcx*4]
    mov [rsp+.i], rcx
    call printf
    mov rcx, [rsp+.i] ; register is volatile
    inc rcx
    mov [rsp+.i], rcx
    cmp rcx, [rsp+.size]
    jl .more
    leave
    ret

; x = min(array,size)
min:
    mov eax, [rdi]
    mov rcx, 1
.more:
    mov r8d, [rdi+rcx*4]
    cmp r8d, eax
    cmovl eax, r8d
    inc rcx
    cmp rcx, rsi
    jl .more
    ret