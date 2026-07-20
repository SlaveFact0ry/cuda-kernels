// v3_register_tile.cu — as in v2, stage BM x BK and BK x BN tiles of A and B
// into __shared__ memory, but now each thread owns an 8x8 (TM x TN)
// micro-tile of C held in registers instead of a single element. Each
// shared-memory read is pulled once into regM/regN and then reused across
// TM*TN FMAs, raising arithmetic intensity per byte of smem traffic.
#include "common.cuh"
#include "gemm.h"

#define BN 128
#define BM 128
#define BK 8
#define TM 8
#define TN 8

__global__ void register_tile(int M, int N, int K, float alpha, float beta,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C){

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int tid = blockDim.x * ty + tx;
  int strideA = (blockDim.x * blockDim.y)/BK;
  int strideB = (blockDim.x * blockDim.y)/BN;
  int innerRowA = tid / BK; int innerColA = tid % BK;
  int innerRowB = tid / BN; int innerColB = tid % BN;

  __shared__ float As[BM][BK], Bs[BK][BN];

  float acc[TM][TN] = {0.f};
  float regM[TM], regN[TN];

  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  for (int t = 0; t < K; t += BK) {
    for(int off = 0; off < BM; off += strideA) As[innerRowA + off][innerColA] = A[ (innerRowA + off)* K + innerColA];
    for(int off = 0; off < BK; off += strideB) Bs[innerRowB +off][innerColB] = B[ (innerRowB + off) * N + innerColB ];
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) regM[i] = As[ty*TM + i][k];
      for (int j = 0; j < TN; ++j) regN[j] = Bs[k][tx*TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
    A += BK;
    B += BK * N;

  }
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      int r = ty*TM + i, c = tx*TN + j;
      C[r*N + c] = alpha*acc[i][j] + beta*C[r*N + c];
    }
}

bool launch_v3_regtile(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
  dim3 block(( BM / TM ) , ( BN / TN )) ;
  dim3 grid(( N + BN - 1) / BN, ( M + BM - 1) / BM ) ;
  register_tile<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  return true;
}
