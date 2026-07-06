// v0_cublas.cu — the speed-of-light (SoL) reference.
// cuBLAS is column-major, but our buffers are ROW-major. To compute the
// row-major product C(MxN) = A(MxK) * B(KxN) we exploit that a row-major
// array is the column-major view of its transpose, and ask cuBLAS for:
//
//     C^T (NxM, col-major)  =  B^T(NxK) * A^T(KxM)
//
// which, written in cuBLAS's column-major convention with NO explicit
// transposes, is exactly:
//
//     cublasSgemm(h, N, N, /*m=*/N, /*n=*/M, /*k=*/K,
//                 alpha, B, ldb=N, A, lda=K, beta, C, ldc=N)
//
// The bytes left in C are the correct row-major MxN result. This is the
// standard idiom; keep it isolated here so the rest of the repo stays
// row-major and easy to reason about.
#include "common.cuh"

// Returns elapsed ms for the timed region (avg over `iters` after `warmup`).
float run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                 float beta, const float* dA, const float* dB, float* dC,
                 int warmup, int iters) {
  for (int i = 0; i < warmup; ++i)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             dB, N, dA, K, &beta, dC, N));
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t; t.start();
  for (int i = 0; i < iters; ++i)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             dB, N, dA, K, &beta, dC, N));
  return t.stop_ms() / iters;
}
