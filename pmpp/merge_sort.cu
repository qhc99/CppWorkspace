

#include "utils.h"
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
constexpr int TILE_SIZE = 256;

__device__ int co_rank_circular(int k, int* A, int m, int* B, int n, int A_S_start, int B_S_start)
{
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = (k - n) < 0 ? 0 : k - n;
    int j_low = (k - m) < 0 ? 0 : k - m;
    int delta {};
    bool active = true;
    while (active) {
        int i_cir = (A_S_start + i) % TILE_SIZE;
        int i_m_1_cir = (A_S_start + i - 1) % TILE_SIZE;
        int j_cir = (B_S_start + j) % TILE_SIZE;
        int j_m_1_cir = (B_S_start + i - 1) % TILE_SIZE;
        if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
            delta = (i - i_low + 1) / 2;
            j_low = j;
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
            delta = (j - j_low + 1) / 2;
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__device__ void merge_sequential_circular(int* A, int m, int* B, int n, int* C, int A_S_start, int B_S_start)
{
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < m) && (j < n)) {
        int i_cir = (A_S_start + i) % TILE_SIZE;
        int j_cir = (B_S_start + j) % TILE_SIZE;
        if (A[i_cir] <= B[j_cir]) {
            C[k++] = A[i_cir];
            i++;
        } else {
            C[k++] = B[j_cir];
            j++;
        }
    }
    if (i == m) {
        for (; j < n; j++) {
            int j_cir = (B_S_start + j) % TILE_SIZE;
            C[k++] = B[j_cir];
        }
    } else {
        for (; i < n; i++) {
            int i_cir = (A_S_start + i) % TILE_SIZE;
            C[k++] = A[i_cir];
        }
    }
}

__global__ void mergeTileKernel(int* A, int m, int* B, int n, int* C)
{
    __shared__ int sharedAB[TILE_SIZE * 2];
    int* A_S = &sharedAB[0];
    int* B_S = &sharedAB[TILE_SIZE];
    int C_curr = static_cast<int>(blockIdx.x) * static_cast<int>(ceil((static_cast<double>(m + n) / gridDim.x)));
    int C_next = min((m + n), (static_cast<int>(blockIdx.x + 1) * static_cast<int>(ceil((static_cast<double>(m + n) / gridDim.x)))));

    if (threadIdx.x == 0) {
        A_S[0] = co_rank_circular(C_curr, A, m, B, n, 0, 0);
        A_S[1] = co_rank_circular(C_next, A, m, B, n, 0, 0);
    }
    __syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(1.0 * C_length / TILE_SIZE);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = TILE_SIZE;
    int B_S_consumed = TILE_SIZE;

    int A_loaded = 0;
    int B_loaded = 0;

    while (counter < total_iteration) {
        for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_loaded && i + threadIdx.x < A_S_consumed) {
                A_S[(A_S_start + (TILE_SIZE - A_S_consumed) + i + threadIdx.x) % TILE_SIZE] = A[A_curr + A_loaded + i + threadIdx.x];
            }
        }
        A_loaded += min(A_S_consumed, A_length - A_loaded);

        for (int i = 0; i < B_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_loaded && i + threadIdx.x < B_S_consumed) {
                B_S[(B_S_start + (TILE_SIZE - B_S_consumed) + i + threadIdx.x) % TILE_SIZE] = B[B_curr + B_loaded + i + threadIdx.x];
            }
        }
        B_loaded += min(B_S_consumed, A_length - B_loaded);

        __syncthreads();

        int c_curr = threadIdx.x * (TILE_SIZE / blockDim.x);
        int c_next = (threadIdx.x + 1) * (TILE_SIZE / blockDim.x);

        c_curr = (c_curr <= (C_length - C_completed)) ? c_curr : (C_length - C_completed);
        c_next = (c_next <= (C_length - C_completed)) ? c_next : (C_length - C_completed);

        int a_curr = co_rank_circular(c_curr, A_S, min(TILE_SIZE, A_length - A_consumed),
            B_S, min(TILE_SIZE, B_length - B_consumed),
            A_S_start, B_S_start);
        int b_curr = c_curr - a_curr;

        int a_next = co_rank_circular(c_next, A_S, min(TILE_SIZE, A_length - A_consumed),
            B_S, min(TILE_SIZE, B_length - B_consumed),
            A_S_start, B_S_start);
        int b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next - a_curr, B_S, b_next - b_curr,
            C + C_curr + c_curr + C_completed,
            A_S_start + a_curr, B_S_start + b_curr);

        counter++;

        A_S_consumed = co_rank_circular(min(TILE_SIZE, C_length - C_completed),
            A_S, min(TILE_SIZE, A_length - A_consumed),
            B_S, min(TILE_SIZE, B_length - B_consumed),
            A_S_start, B_S_start);

        B_S_consumed = min(TILE_SIZE, C_length - C_completed) - A_S_consumed;

        A_consumed += A_S_consumed;
        C_completed += min(TILE_SIZE, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % TILE_SIZE;
        B_S_start = (B_S_start + B_S_consumed) % TILE_SIZE;

        __syncthreads();
    }
}

void mergeTiles(int* A, int m, int* B, int n, int* C)
{
    int* A_d { nullptr };
    int* B_d { nullptr };
    int* C_d { nullptr };
    checkCudaError(cudaMalloc(&A_d, m * sizeof(int)));
    checkCudaError(cudaMalloc(&B_d, n * sizeof(int)));
    checkCudaError(cudaMalloc(&C_d, (m + n) * sizeof(int)));
    checkCudaError(cudaMemcpy(A_d, A, m * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(B_d, B, n * sizeof(int), cudaMemcpyHostToDevice));

    mergeTileKernel<<<max(1,static_cast<unsigned int>(floor((m+n)/(4.*TILE_SIZE)))), 64>>>(A_d, m, B_d, n, C_d);

    checkCudaLastError();

    checkCudaError(cudaMemcpy(C, C_d, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(A_d));
    checkCudaError(cudaFree(B_d));
    checkCudaError(cudaFree(C_d));
}

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-pro-bounds-pointer-arithmetic,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index)
