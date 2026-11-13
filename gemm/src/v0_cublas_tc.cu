// v0_cublas_tc.cu — the tensor-core speed-of-light references, TF32 and BF16.
// Same row-major trick as v0_cublas.cu: row-major C(MxN) = A(MxK)*B(KxN) is
// obtained from cuBLAS (column-major) as C^T = B^T * A^T, i.e.
//     cublasGemmEx(h, N, N, /*m=*/N, /*n=*/M, /*k=*/K, ..., B, ldb=N, A, lda=K, ..., C, ldc=N)
// with no explicit transposes.
#include "common.cuh"
#include <cuda_bf16.h>

// TF32: storage stays FP32 (TF32 is a compute mode, not a storage type);
// the inputs are expected to already be pre-rounded to TF32-representable
// values (see round_to_tf32 in gemm_tc.h) so this is an apples-to-apples
// compute-only comparison against our own TF32 kernels.
float run_cublas_tf32(cublasHandle_t handle, int M, int N, int K, float alpha,
                      float beta, const float* dA, const float* dB, float* dC,
                      int warmup, int iters) {
  for (int i = 0; i < warmup; ++i)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              dB, CUDA_R_32F, N, dA, CUDA_R_32F, K, &beta,
                              dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32,
                              CUBLAS_GEMM_DEFAULT));
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t; t.start();
  for (int i = 0; i < iters; ++i)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              dB, CUDA_R_32F, N, dA, CUDA_R_32F, K, &beta,
                              dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F_FAST_TF32,
                              CUBLAS_GEMM_DEFAULT));
  return t.stop_ms() / iters;
}

// BF16: storage for A/B is __nv_bfloat16, accumulate and output in FP32.
float run_cublas_bf16(cublasHandle_t handle, int M, int N, int K, float alpha,
                      float beta, const __nv_bfloat16* dA,
                      const __nv_bfloat16* dB, float* dC, int warmup,
                      int iters) {
  for (int i = 0; i < warmup; ++i)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, K, &beta,
                              dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t; t.start();
  for (int i = 0; i < iters; ++i)
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, K, &beta,
                              dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
  return t.stop_ms() / iters;
}
