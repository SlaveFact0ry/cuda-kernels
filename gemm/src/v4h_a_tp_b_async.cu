#include <cuda_pipeline.h>
#include "common.cuh"
#include "gemm.h"

#define BN 128
#define BM 128
#define BK 8
#define TM 8
#define TN 8

__global__ void v4h_a_tp_b_async(int M, int N, int K, float alpha, float beta,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C){

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int innerRowA = tid / (BK/4); int innerColA = tid % (BK/4);
    int innerRowB = tid / (BN/4); int innerColB = tid % (BN/4);
    int col0      = innerColA * 4;
    int col0B     = innerColB * 4;
    float pf_a[4];

    __shared__ float As[2][BK][BM], Bs[2][BK][BN];

    float regM[TM], regN[TN];
    float acc[TM][TN] = {0.f};

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // ---- prologue: tile 0 -> buffer 0 ---
    *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0]);
    As[0][col0][innerRowA]= pf_a[0];
    As[0][col0+1][innerRowA] = pf_a[1];
    As[0][col0+2][innerRowA] = pf_a[2];
    As[0][col0+3][innerRowA] = pf_a[3];
    __pipeline_memcpy_async(&Bs[0][innerRowB][col0B], &B[innerRowB * N + col0B ], 16);  // (1) 16B 비동기 복사 '주문'
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        // (1) prefetch NEXT tile into registers — LDG issued BEFORE compute
        int nxt = t + BK;
        if (nxt < K) {
            *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0 + nxt]);
            __pipeline_memcpy_async(&Bs[cur^1][innerRowB][col0B], &B[(innerRowB + nxt)* N +col0B], 16);  // (1) 16B 비동기 복사 '주문'
            __pipeline_commit();      // 다음 스테이지 봉인
        }
        // (2) compute on CURRENT buffer — the ENTIRE k-loop
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            *reinterpret_cast<float4*>(&regM[0]) = *reinterpret_cast<float4*>(&As[cur][k][threadIdx.y*TM]);
            *reinterpret_cast<float4*>(&regM[4]) = *reinterpret_cast<float4*>(&As[cur][k][threadIdx.y*TM + 4]);
            *reinterpret_cast<float4*>(&regN[0]) = *reinterpret_cast<float4*>(&Bs[cur][k][threadIdx.x*TN]);
            *reinterpret_cast<float4*>(&regN[4]) = *reinterpret_cast<float4*>(&Bs[cur][k][threadIdx.x*TN+4]);
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        if (nxt < K) {
            As[cur^1][col0][innerRowA] = pf_a[0];
            As[cur^1][col0+1][innerRowA] = pf_a[1];
            As[cur^1][col0+2][innerRowA] = pf_a[2];
            As[cur^1][col0+3][innerRowA] = pf_a[3];
            __pipeline_wait_prior(0);
            __syncthreads();
            cur ^= 1;                                     // 버퍼 교체
        }
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int r = threadIdx.y*TM + i, c = threadIdx.x*TN + j;
            C[r*N + c] = alpha*acc[i][j] + beta*C[r*N + c];
        }
}

bool launch_v4h_a_tp_b_async(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
      if (M % BM || N % BN || K % BK) {
        fprintf(stderr,
            "v4h_a_tp_b_async: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
    }
    dim3 block(( BN / TN ),( BM / TM )) ;
    dim3 grid(N / BN, M / BM);         // exact division now — no false ceil-div promise
    v4h_a_tp_b_async<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
