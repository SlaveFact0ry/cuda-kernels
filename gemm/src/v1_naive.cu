// v1_naive.cu — one thread computes one C element. No reuse: every thread
// streams a full row of A and a full column of B from global memory. This is
// the deliberately-bad baseline. Expect it to be MEMORY-bound and to sit far
// below the cuBLAS SoL. Your job in v2+ is to close that gap; profile this
// first with NCU and CONFIRM the bottleneck before optimizing.
#include "common.cuh"
#include "gemm.h"

__global__ void naive_sgemm(int M, int N, int K, float alpha, float beta,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..M-1
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..N-1
  if (row >= M || col >= N) return;

  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];          // row-major indexing

  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

bool launch_v1_naive(int M, int N, int K, float alpha, float beta,
                     const float* dA, const float* dB, float* dC) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  naive_sgemm<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  return true;
}
