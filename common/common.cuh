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

// max |a - b| over n elements; used to verify a kernel against the cuBLAS ref
inline double max_abs_diff(const std::vector<float>& a,
                           const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = fmax(m, (double)fabsf(a[i] - b[i]));
  return m;
}

// 2*M*N*K FLOPs for C = A*B (mul + add per inner-product term)
inline double gemm_gflops(int M, int N, int K, float ms) {
  return 2.0 * (double)M * (double)N * (double)K / (ms * 1e6);
}
