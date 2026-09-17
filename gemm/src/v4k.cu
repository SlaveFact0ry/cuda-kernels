#include "common.cuh"
#include "gemm.h"
#define BN 128
#define BM 128
#define BK 8
#define TM 8
#define TN 16
#define WM 64
#define WN 64
#define WNITER 1
#define NUM_THREADS 128
constexpr uint WARPSIZE  = 32;
constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;      // 4
constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER); // 4096/4096 = 1
constexpr uint WSUBM  = WM / WMITER;   // 64
constexpr uint WSUBN  = WN / WNITER;   // 64

static_assert(BM % WM == 0 && BN % WN == 0);
static_assert((BM / WM) * (BN / WN) == NUM_WARPS);      // 2*2 == 4
static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
static_assert(WSUBM % TM == 0 && WSUBN % TN == 0);       // 64%8==0, 64%16==0
static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE);  // 8*4 == 32

__device__ __forceinline__ int swz_chunk(int col) {
    int chunk = col >> 2;              // 0..31, float4-chunk index
    chunk ^= (chunk >> 3) & 7;         // self-referential permutation
    return (chunk << 2) | (col & 3);   // reconstruct — stays 4-aligned for float4
}

__global__ void v4k_wide_tile(int M, int N, int K, float alpha, float beta,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C){

    int tid = threadIdx.x;
    // ---- loader coords: 2-pass (NUM_THREADS=128 covers BM/BK tiles in 2 passes) ----
    const uint innerRowA0 = tid / (BK / 4), innerColA = tid % (BK / 4);   // BK/4=2 -> innerRowA0 range 0..63
    const uint innerRowB0 = tid / (BN / 4), innerColB = tid % (BN / 4);   // BN/4=32 -> innerRowB0 range 0..3
    int col0  = innerColA * 4;
    int col0B = innerColB * 4;
    constexpr uint A_ROW_STRIDE = NUM_THREADS / (BK / 4);   // 128/2 = 64
    constexpr uint B_ROW_STRIDE = NUM_THREADS / (BN / 4);   // 128/32 = 4
    constexpr uint A_PASSES = BM / A_ROW_STRIDE;             // 128/64 = 2
    constexpr uint B_PASSES = BK / B_ROW_STRIDE;             // 8/4 = 2

    // ---- warptile coords: separate system, do NOT mix with the above ----
    const uint warpIdx = tid / WARPSIZE;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);
    const uint tInWarp = tid % WARPSIZE;
    const uint threadRowInWarp = tInWarp / (WSUBN / TN);   // tInWarp/4, range 0..7
    const uint threadColInWarp = tInWarp % (WSUBN / TN);   // tInWarp%4, range 0..3

    __shared__ float As[2][BK][BM], Bs[2][BK][BN];
    float pf_a[A_PASSES][4], pf_b[B_PASSES][4];

    float acc[WMITER * TM][WNITER * TN] = {0.f};
    __align__(16) float regM[WMITER * TM];
    __align__(16) float regN[WNITER * TN];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // ---- prologue: tile 0 -> buffer 0 ----
    #pragma unroll
    for (uint p = 0; p < A_PASSES; ++p) {
        uint row = innerRowA0 + p * A_ROW_STRIDE;
        *reinterpret_cast<float4*>(&pf_a[p]) = *reinterpret_cast<const float4*>(&A[row * K + col0]);
        As[0][col0][row]   = pf_a[p][0];
        As[0][col0+1][row] = pf_a[p][1];
        As[0][col0+2][row] = pf_a[p][2];
        As[0][col0+3][row] = pf_a[p][3];
    }
    #pragma unroll
    for (uint p = 0; p < B_PASSES; ++p) {
        uint row = innerRowB0 + p * B_ROW_STRIDE;
        *reinterpret_cast<float4*>(&Bs[0][row][swz_chunk(col0B)]) = *reinterpret_cast<const float4*>(&B[row * N + col0B]);
    }
    __syncthreads();
    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool nxt = t + BK < K;
        if (nxt) {
            #pragma unroll
            for (uint p = 0; p < A_PASSES; ++p) {
                uint row = innerRowA0 + p * A_ROW_STRIDE;
                *reinterpret_cast<float4*>(&pf_a[p]) = *reinterpret_cast<const float4*>(&A[row * K + col0 + BK]);
            }
            #pragma unroll
            for (uint p = 0; p < B_PASSES; ++p) {
                uint row = innerRowB0 + p * B_ROW_STRIDE;
                *reinterpret_cast<float4*>(&pf_b[p]) = *reinterpret_cast<const float4*>(&B[(row + BK) * N + col0B]);
            }
        }

        #pragma unroll
        for (uint k = 0; k < BK; ++k) {
            #pragma unroll
            for (uint wr = 0; wr < WMITER; ++wr) {
                *reinterpret_cast<float4*>(&regM[wr * TM])     = *reinterpret_cast<float4*>(&As[cur][k][warpRow * WM + wr * WSUBM + threadRowInWarp * TM]);
                *reinterpret_cast<float4*>(&regM[wr * TM + 4]) = *reinterpret_cast<float4*>(&As[cur][k][warpRow * WM + wr * WSUBM + threadRowInWarp * TM + 4]);
            }
            #pragma unroll
            for (uint wc = 0; wc < WNITER; ++wc) {
                uint base = warpCol * WN + wc * WSUBN + threadColInWarp * TN;
                *reinterpret_cast<float4*>(&regN[wc * TN])      = *reinterpret_cast<float4*>(&Bs[cur][k][swz_chunk(base)]);
                *reinterpret_cast<float4*>(&regN[wc * TN + 4])  = *reinterpret_cast<float4*>(&Bs[cur][k][swz_chunk(base + 4)]);
                *reinterpret_cast<float4*>(&regN[wc * TN + 8])  = *reinterpret_cast<float4*>(&Bs[cur][k][swz_chunk(base + 8)]);
                *reinterpret_cast<float4*>(&regN[wc * TN + 12]) = *reinterpret_cast<float4*>(&Bs[cur][k][swz_chunk(base + 12)]);
            }
            #pragma unroll
            for (uint m = 0; m < WMITER * TM; ++m)
                #pragma unroll
                for (uint n = 0; n < WNITER * TN; ++n)
                    acc[m][n] += regM[m] * regN[n];
        }

        if (nxt) {
            #pragma unroll
            for (uint p = 0; p < A_PASSES; ++p) {
                uint row = innerRowA0 + p * A_ROW_STRIDE;
                As[cur^1][col0][row]   = pf_a[p][0];
                As[cur^1][col0+1][row] = pf_a[p][1];
                As[cur^1][col0+2][row] = pf_a[p][2];
                As[cur^1][col0+3][row] = pf_a[p][3];
            }
            #pragma unroll
            for (uint p = 0; p < B_PASSES; ++p) {
                uint row = innerRowB0 + p * B_ROW_STRIDE;
                *reinterpret_cast<float4*>(&Bs[cur^1][row][swz_chunk(col0B)]) = *reinterpret_cast<const float4*>(&pf_b[p]);
            }
            __syncthreads();
            cur ^= 1;
        }
        A += BK;
        B += BK * N;
    }
    #pragma unroll
    for (uint m = 0; m < WMITER * TM; ++m) {
        const uint r = warpRow * WM + (m / TM) * WSUBM + threadRowInWarp * TM + (m % TM);
        #pragma unroll
        for (uint n = 0; n < WNITER * TN; ++n) {
            const uint c = warpCol * WN + (n / TN) * WSUBN + threadColInWarp * TN + (n % TN);
            C[r * N + c] = alpha * acc[m][n] + beta * C[r * N + c];
        }
    }
}

bool launch_v4k(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr,
            "v4k: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
    }
    dim3 block(NUM_THREADS);
    dim3 grid(div_up(N, BN), div_up(M, BM));   // exact by the guard above
    v4k_wide_tile<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
