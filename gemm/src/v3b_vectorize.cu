#include "common.cuh"
#include "gemm.h"

#define BN 128
#define BM 128
#define BK 8
#define TM 8
#define TN 8

__global__ void register_tile_vec(int M, int N, int K, float alpha, float beta,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C){

  int tid = blockDim.x * threadIdx.y + threadIdx.x;
  int strideA = (blockDim.x * blockDim.y)/BK;
  int strideB = (blockDim.x * blockDim.y)/BN;
  int innerRowA = tid / BK; int innerColA = tid % BK;
  int innerRowB = tid / BN; int innerColB = tid % BN;

  __shared__ float As[BK][BM], Bs[BK][BN];

  float acc[TM][TN] = {0.f};
  float regM[TM], regN[TN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  for (int t = 0; t < K; t += BK) {
    for(int off = 0; off < BM; off += strideA) As[innerColA][innerRowA + off] = A[ (innerRowA + off)* K + innerColA];
    for(int off = 0; off < BK; off += strideB) Bs[innerRowB +off][innerColB] = B[ (innerRowB + off) * N + innerColB ];
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
        reinterpret_cast<float4*>(regM)[0] = reinterpret_cast<const float4*>(&As[k][threadIdx.y*TM])[0];
        reinterpret_cast<float4*>(regM)[1] = reinterpret_cast<const float4*>(&As[k][threadIdx.y*TM])[1];
        reinterpret_cast<float4*>(regN)[0] = reinterpret_cast<const float4*>(&Bs[k][threadIdx.x*TN])[0];
        reinterpret_cast<float4*>(regN)[1] = reinterpret_cast<const float4*>(&Bs[k][threadIdx.x*TN])[1];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
    A += BK;
    B += BK * N;
  }
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      int r = threadIdx.y*TM + i, c = threadIdx.x*TN + j;
      C[r*N + c] = alpha*acc[i][j] + beta*C[r*N + c];
    }
}

bool launch_v3b_vectorize(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
  if (M % BM || N % BN || K % BK) {
    fprintf(stderr,
            "v3b_vectorize: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
  }
  dim3 block(( BM / TM ) , ( BN / TN )) ;
  dim3 grid(N / BN, M / BM);         // exact division now — no false ceil-div promise
  register_tile_vec<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  return true;
}
