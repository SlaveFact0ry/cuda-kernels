#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---- error checking ---------------------------------------------------------
#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__,        \
              __LINE__, cudaGetErrorString(_e));                               \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(expr)                                                      \
  do {                                                                          \
    cublasStatus_t _s = (expr);                                                 \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuBLAS error %s at %s:%d (status %d)\n", #expr,          \
              __FILE__, __LINE__, (int)_s);                                     \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---- launch geometry --------------------------------------------------------
// ceil(a / b). Use it to turn a PROBLEM extent into a block count, where the
// divisor is the number of elements one block covers along that axis -- that is
// blockDim for a one-element-per-thread kernel, and the tile width (BM/BN) for
// a register-tiled one. Dividing an already-tiled count by blockDim mixes the
// two levels and undercounts blocks.
__host__ __device__ constexpr int div_up(int a, int b) {   // constexpr implies inline
  return (a + b - 1) / b;
}

// ---- GPU event timer --------------------------------------------------------
struct GpuTimer {
  cudaEvent_t a, b;
  GpuTimer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
  ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start() { CUDA_CHECK(cudaEventRecord(a)); }
  float stop_ms() {                          // milliseconds for the recorded range
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  }
};

// ---- host helpers -----------------------------------------------------------
inline void fill_random(std::vector<float>& v, unsigned seed = 1234) {
  srand(seed);
  for (auto& x : v) x = (float)rand() / (float)RAND_MAX * 2.f - 1.f;  // [-1, 1]
}

// max relative error of out vs ref, denominator floored at 1 to stay stable
// near zero; used to verify a kernel against the cuBLAS reference. Relative
// (not absolute) so the tolerance holds as K grows and per-element error
// accumulates.
inline double max_rel_diff(const std::vector<float>& out,
                           const std::vector<float>& ref) {
  double m = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    double denom = fmax(1.0, (double)fabsf(ref[i]));
    m = fmax(m, (double)fabsf(out[i] - ref[i]) / denom);
  }
  return m;
}

// 2*M*N*K FLOPs for C = A*B (mul + add per inner-product term)
inline double gemm_gflops(int M, int N, int K, float ms) {
  return 2.0 * (double)M * (double)N * (double)K / (ms * 1e6);
}
