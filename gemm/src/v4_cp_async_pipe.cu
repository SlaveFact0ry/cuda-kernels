// v4_cp_async_pipe.cu — YOUR RUNG. *This is the rung Ampere unlocks.*
//
// On the old Turing card this step was a "manual" double buffer because
// cp.async did not exist. On the 3090 (sm_80+) you have asynchronous
// global->shared copies, so v4 becomes a real SOFTWARE PIPELINE — the same
// idea CUTLASS uses.
//
// Two combinable ideas on top of v3:
//   (a) VECTORIZED transfers: move A/B tiles in 16-byte chunks (float4). This
//       maps perfectly onto cp.async's 16-byte mode.
//   (b) cp.async MULTI-STAGE PIPELINE: issue the copy for tile (k+1)[..k+S-1]
//       into separate shared-memory buffers while the MMA math for tile k runs,
//       then wait only on the stage you are about to consume. Hides global
//       latency behind compute.
//
// Primitives (#include <cuda_pipeline.h>):
//   __pipeline_memcpy_async(smem_ptr, gmem_ptr, 16);  // 16B async copy
//   __pipeline_commit();                              // close a stage
//   __pipeline_wait_prior(S-1);                       // wait until <=S-1 in flight
//   __syncthreads();                                  // before reading that smem
//
// Shape: keep v3's BM=128,BN=128,BK=8 micro-tiling, but allocate the A/B smem
// tiles S times (S = 2..4 stages) and round-robin the buffer index.
//
// Watch with NCU (this is the payoff measurement):
//   - "long scoreboard" / memory stalls should DROP vs v3 (latency now hidden);
//   - DRAM throughput as % of peak (~936 GB/s on the 3090);
//   - did issue efficiency / achieved occupancy hold while latency hid?
//   - Ampere note: cp.async.cg bypasses L1 and lands in shared via L2; reason
//     about whether that helps your reuse pattern.
//
// This rung is the single most "modern GPU" thing in the repo — make the
// before/after stall breakdown the centerpiece of docs/notes.md.
#include <cuda_pipeline.h>
#include "common.cuh"
#include "gemm.h"

#define BN 128
#define BM 128
#define BK 8
#define TM 8
#define TN 8

__global__ void cp_async_pipe(int M, int N, int K, float alpha, float beta,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C){

  int tid = blockDim.x * threadIdx.y + threadIdx.x;
  int innerRowA = tid / (BK/4); int innerColA = tid % (BK/4);
  int innerRowB = tid / (BN/4); int innerColB = tid % (BN/4);
  int col0      = innerColA * 4;
  int col0B     = innerColB * 4;

  __shared__ float As[2][BM][BK], Bs[2][BK][BN];

  float regM[TM], regN[TN];
  float acc[TM][TN] = {0.f};

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // ---- prologue: tile 0 -> buffer 0 ----
  // *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0]);
  // *reinterpret_cast<float4*>(&pf_b) = *reinterpret_cast<const float4*>(&B[innerRowB * N + col0B ]);
  // *reinterpret_cast<float4*>(&As[0][innerRowA][innerColA]) = *reinterpret_cast<const float4*>(&pf_a);
  // *reinterpret_cast<float4*>(&Bs[0][innerRowB][innerColB]) = *reinterpret_cast<const float4*>(&pf_b);
  __pipeline_memcpy_async(&As[0][innerRowA][col0], &A[innerRowA* K +col0], 16);  // (1) 16B 비동기 복사 '주문'
  __pipeline_memcpy_async(&Bs[0][innerRowB][col0B], &B[innerRowB * N + col0B ], 16);  // (1) 16B 비동기 복사 '주문'
  __pipeline_commit();                              // (2) 여기까지 주문을 한 '스테이지'로 묶음
  __pipeline_wait_prior(0);                         // (3) 아직 안 끝난 스테이지가 N개 이하가 될 때까지 대기
  __syncthreads();                                  // (4) 그 smem을 '읽기 전에' 워프 동기화

    // __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        // (1) prefetch NEXT tile into registers — LDG issued BEFORE compute
        int nxt = t + BK;
        if (nxt < K) {
          __pipeline_memcpy_async(&As[cur^1][innerRowA][col0], &A[innerRowA* K +col0 + nxt], 16);  // (1) 16B 비동기 복사 '주문'
          __pipeline_memcpy_async(&Bs[cur^1][innerRowB][col0B], &B[(innerRowB + nxt)* N +col0B], 16);  // (1) 16B 비동기 복사 '주문'
          __pipeline_commit();      // 다음 스테이지 봉인
            // *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0 + nxt]);
            // *reinterpret_cast<float4*>(&pf_b) = *reinterpret_cast<const float4*>(&B[(innerRowB+nxt) * N  + col0B]);
        }
        // (2) compute on CURRENT buffer — the ENTIRE k-loop
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regM[i] = As[cur][threadIdx.y*TM + i][k];
            *reinterpret_cast<float4*>(&regN[0]) = *reinterpret_cast<float4*>(&Bs[cur][k][threadIdx.x*TM]);
            *reinterpret_cast<float4*>(&regN[4]) = *reinterpret_cast<float4*>(&Bs[cur][k][threadIdx.x*TN+4]);
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        // (3) store prefetched tile into the OTHER buffer — ONE sync, THEN swap
        if (nxt < K) {
            // *reinterpret_cast<float4*>(&As[cur^1][innerRowA][innerColA]) = *reinterpret_cast<const float4*>(&pf_a);
            // *reinterpret_cast<float4*>(&Bs[cur^1][innerRowB][innerColB]) = *reinterpret_cast<const float4*>(&pf_b);
          __pipeline_wait_prior(0);   // in-flight 0개 될 때까지 (2-stage)
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


bool launch_v4_cp_async(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
      if (M % BM || N % BN || K % BK) {
        fprintf(stderr,
            "v4_cp_async_pipe: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
    }
    dim3 block(( BM / TM ) , ( BN / TN )) ;
    dim3 grid(N / BN, M / BM);         // exact division now — no false ceil-div promise
    cp_async_pipe<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
