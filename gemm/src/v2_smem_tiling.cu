// v2_smem_tiling.cu — stage BM x BK and BK x BN tiles of A and B into
// __shared__ memory so each loaded element is reused by the whole block
// instead of re-fetched per thread. Inner-loop reads are bank-conflict-free:
// As[ty][k] is a broadcast (same address for every tx in the warp), Bs[k][tx]
// hits consecutive banks across tx. NCU checklist + Ampere smem notes:
// ../docs/gemm-notes.md.

#include "common.cuh"
#include "gemm.h"


#define BM 32
#define BN 32
#define BK 32


__global__ void smem_sgemm(int M, int N, int K, float alpha, float beta,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C){
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    float acc = 0.f;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockRow * BM + ty;
    int col = blockCol * BN + tx;

    __shared__ float As[BM][BK], Bs[BK][BN];

    for (int t = 0; t < K; t += BK) {

        As[ty][tx] = A[row * K + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < BK; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();

    }
    C[row*N+col] = alpha*acc + beta*C[row*N+col];

  }

bool launch_v2_smem(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
  if (M % BM || N % BN || K % BK) {
    fprintf(stderr,
            "v2_smem_tiling: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
  }
  dim3 block(BN, BM);
  dim3 grid(N / BN, M / BM);         // exact division now — no false ceil-div promise
  smem_sgemm<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  return true;
}
