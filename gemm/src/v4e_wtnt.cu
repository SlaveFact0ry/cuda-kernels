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

__global__ void warptiling_notranspose(int M, int N, int K, float alpha, float beta,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C){

    int tid = threadIdx.x;
    // int innerRowA = tid / (BK/4); int innerColA = tid % (BK/4);
    // int innerRowB = tid / (BN/4); int innerColB = tid % (BN/4);
    // ---- loader coords: identical to v3b, keep flat tid ----
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

    __shared__ float As[BK*BM], Bs[BK*BN];

    float acc[WMITER * TM][WNITER * TN] = {0.f};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    for (int t = 0; t < K; t += BK) {
        // float4 tmp = *reinterpret_cast<const float4*>(&A[innerRowA* K + col0]);
        // As[col0 * BM + innerRowA] = tmp.x;
        // As[(col0+1) * BM + innerRowA] = tmp.y;
        // As[(col0+2) * BM + innerRowA] = tmp.z;
        // As[(col0+3) * BM + innerRowA] = tmp.w;
        *reinterpret_cast<float4*>(&As[innerRowA * BK + col0]) = *reinterpret_cast<const float4*>(&A[innerRowA * K + col0 ]);
        *reinterpret_cast<float4*>(&Bs[innerRowB * BN + col0B]) = *reinterpret_cast<const float4*>(&B[innerRowB * N + col0B ]);
        __syncthreads();

        #pragma unroll
        for (uint k = 0; k < BK; ++k) {
            #pragma unroll
            for (uint wr = 0; wr < WMITER; ++wr)
                #pragma unroll
                for (uint i = 0; i < TM; ++i)
                    // regM[wr * TM + i] = As[k * BM + warpRow * WM + wr * WSUBM + threadRowInWarp * TM + i];
                    regM[wr * TM + i] = As[(warpRow * WM + wr * WSUBM + threadRowInWarp * TM + i) * BK + k];

            #pragma unroll
            for (uint wc = 0; wc < WNITER; ++wc)
                #pragma unroll
                for (uint i = 0; i < TN; ++i)
                    regN[wc * TN + i] = Bs[k * BN + warpCol * WN + wc * WSUBN + threadColInWarp * TN + i];

            #pragma unroll
            for (uint m = 0; m < WMITER * TM; ++m)
                #pragma unroll
                for (uint n = 0; n < WNITER * TN; ++n)
                    acc[m][n] += regM[m] * regN[n];
        }
        __syncthreads();
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

bool launch_v4e_wtnt(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr,"v4e_wtnt: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
    }
    dim3 block(NUM_THREADS);              // x -> N (col), y -> M (row)
    dim3 grid(div_up(N, BN), div_up(M, BM));   // exact by the guard above
    warptiling_notranspose<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//   10.633    12926.0     62.9