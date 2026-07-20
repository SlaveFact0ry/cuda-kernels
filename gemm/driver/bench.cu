// bench.cu — problem setup, cuBLAS speed-of-light baseline, then for each
// registered version: verify against cuBLAS and time it. Prints a table you can
// paste straight into the README. Usage:  ./bench [M] [N] [K] [iters]
#include "common.cuh"
#include "gemm.h"

// defined in v0_cublas.cu
float run_cublas(cublasHandle_t, int, int, int, float, float, const float*,
                 const float*, float*, int, int);

static float time_launcher(GemmLauncher fn, int M, int N, int K, float alpha,
                           float beta, const float* dA, const float* dB,
                           float* dC, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn(M, N, K, alpha, beta, dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t; t.start();
  for (int i = 0; i < iters; ++i) fn(M, N, K, alpha, beta, dA, dB, dC);
  return t.stop_ms() / iters;
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int N = argc > 2 ? atoi(argv[2]) : 2048;
  int K = argc > 3 ? atoi(argv[3]) : 2048;
  int iters  = argc > 4 ? atoi(argv[4]) : 50;
  int warmup = 10;
  float alpha = 1.f, beta = 0.f;

  printf("GEMM  M=%d N=%d K=%d  (row-major, fp32)  iters=%d\n", M, N, K, iters);

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  std::vector<float> h_ref((size_t)M * N), h_out((size_t)M * N);
  fill_random(hA, 1); fill_random(hB, 2);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, h_ref.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));

  // ---- speed-of-light reference --------------------------------------------
  CUDA_CHECK(cudaMemset(dC, 0, h_ref.size()*sizeof(float)));
  float ms_ref = run_cublas(handle, M, N, K, alpha, beta, dA, dB, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(h_ref.data(), dC, h_ref.size()*sizeof(float), cudaMemcpyDeviceToHost));
  double sol = gemm_gflops(M, N, K, ms_ref);
  printf("\n%-18s %10s %10s %8s  %s\n", "version", "ms", "GFLOP/s", "%SoL", "verify");
  printf("%-18s %10.3f %10.1f %8s  %s\n", "v0_cublas(SoL)", ms_ref, sol, "100.0", "-");

  // ---- each version ---------------------------------------------------------
  constexpr double kRelTol = 1e-3;  // fp32 vs cuBLAS, calibrated at 8192^3
  int failures = 0;
  int n = 0; const Version* tbl = version_table(n);
  for (int i = 0; i < n; ++i) {
    CUDA_CHECK(cudaMemset(dC, 0, h_ref.size()*sizeof(float)));
    bool ok = tbl[i].fn(M, N, K, alpha, beta, dA, dB, dC);
    if (!ok) { printf("%-18s %10s %10s %8s  %s\n", tbl[i].name, "-", "-", "-", "skip"); continue; }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), dC, h_ref.size()*sizeof(float), cudaMemcpyDeviceToHost));
    double diff = max_rel_diff(h_out, h_ref);
    bool pass = diff < kRelTol;
    if (!pass) ++failures;
    float ms = time_launcher(tbl[i].fn, M, N, K, alpha, beta, dA, dB, dC, warmup, iters);
    double g = gemm_gflops(M, N, K, ms);
    printf("%-18s %10.3f %10.1f %8.1f  maxdiff=%.2e %s\n", tbl[i].name, ms, g,
           100.0 * g / sol, diff, pass ? "PASS" : "FAIL");
  }

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  return failures ? 1 : 0;
}
