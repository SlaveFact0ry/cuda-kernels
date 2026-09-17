#include "common.cuh"
#include "gemm.h"
#define BN 128
#define BM 128
#define BK 8
#define TM 8
#define TN 4
#define WM 64
#define WN 32
#define WNITER 2
#define NUM_THREADS 256
constexpr uint WARPSIZE  = 32;
constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;
constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
constexpr uint WSUBM  = WM / WMITER;
constexpr uint WSUBN  = WN / WNITER;

static_assert(BM % WM == 0 && BN % WN == 0);
static_assert((BM / WM) * (BN / WN) == NUM_WARPS);
static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
static_assert(WSUBM % TM == 0 && WSUBN % TN == 0);
static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE);

__device__ __forceinline__ int swz_chunk(int col) {
    int chunk = col >> 2;              // 0..31, float4-chunk index
    chunk ^= (chunk >> 3) & 7;         // self-referential permutation
    return (chunk << 2) | (col & 3);   // reconstruct — stays 4-aligned for float4
}

__global__ void warptiling_transpose_swizzle(int M, int N, int K, float alpha, float beta,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C){

    int tid = threadIdx.x;
    // ---- loader coords: identical to v4g2, keep flat tid ----
    const uint innerRowA = tid / (BK / 4), innerColA = tid % (BK / 4);
    const uint innerRowB = tid / (BN / 4), innerColB = tid % (BN / 4);
    int col0 = innerColA * 4;
    int col0B = innerColB * 4;

    // ---- warptile coords: separate system, do NOT mix with the above ----
    const uint warpIdx = tid / WARPSIZE;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);
    const uint tInWarp = tid % WARPSIZE;
    const uint threadRowInWarp = tInWarp / (WSUBN / TN);
    const uint threadColInWarp = tInWarp % (WSUBN / TN);

    __shared__ float As[2][BK][BM], Bs[2][BK][BN];
    float pf_a[4], pf_b[4];

    float acc[WMITER * TM][WNITER * TN] = {0.f};
    __align__(16) float regM[WMITER * TM];
    __align__(16) float regN[WNITER * TN];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // ---- prologue: tile 0 -> buffer 0 ----
    *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0]);
    *reinterpret_cast<float4*>(&pf_b) = *reinterpret_cast<const float4*>(&B[innerRowB * N + col0B ]);
    As[0][col0][innerRowA] = pf_a[0];
    As[0][col0+1][innerRowA] = pf_a[1];
    As[0][col0+2][innerRowA] = pf_a[2];
    As[0][col0+3][innerRowA] = pf_a[3];
    *reinterpret_cast<float4*>(&Bs[0][innerRowB][swz_chunk(col0B)]) = *reinterpret_cast<const float4*>(&pf_b);
    __syncthreads();
    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        const bool nxt = t + BK < K;
        if (nxt) {
            *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0 + BK]);
            *reinterpret_cast<float4*>(&pf_b) = *reinterpret_cast<const float4*>(&B[(innerRowB+BK) * N  + col0B]);
        }

        #pragma unroll
        for (uint k = 0; k < BK; ++k) {
            #pragma unroll
            for (uint wr = 0; wr < WMITER; ++wr) {
                *reinterpret_cast<float4*>(&regM[wr * TM]) = *reinterpret_cast<float4*>(&As[cur][k][warpRow * WM + wr * WSUBM + threadRowInWarp * TM]);
                *reinterpret_cast<float4*>(&regM[wr * TM + 4]) = *reinterpret_cast<float4*>(&As[cur][k][warpRow * WM + wr * WSUBM + threadRowInWarp * TM + 4]);
            }

            #pragma unroll
            for (uint wc = 0; wc < WNITER; ++wc) {
                *reinterpret_cast<float4*>(&regN[wc * TN]) = *reinterpret_cast<float4*>(&Bs[cur][k][swz_chunk(warpCol * WN + wc * WSUBN + threadColInWarp * TN)]);
            }

            #pragma unroll
            for (uint m = 0; m < WMITER * TM; ++m)
                #pragma unroll
                for (uint n = 0; n < WNITER * TN; ++n)
                    acc[m][n] += regM[m] * regN[n];
        }
        // (3) store prefetched tile into the OTHER buffer — ONE sync, THEN swap
        if (nxt) {
            As[cur^1][col0][innerRowA] = pf_a[0];
            As[cur^1][col0+1][innerRowA] = pf_a[1];
            As[cur^1][col0+2][innerRowA] = pf_a[2];
            As[cur^1][col0+3][innerRowA] = pf_a[3];
            *reinterpret_cast<float4*>(&Bs[cur^1][innerRowB][swz_chunk(col0B)]) = *reinterpret_cast<const float4*>(&pf_b);
            __syncthreads();
            cur ^= 1;                                        // 버퍼 교체
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

bool launch_v4g3(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr,
            "v4g3: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
    }
    dim3 block(NUM_THREADS);
    dim3 grid(div_up(N, BN), div_up(M, BM));   // exact by the guard above
    warptiling_transpose_swizzle<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
