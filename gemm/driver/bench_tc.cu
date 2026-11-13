// bench_tc.cu — v5-0 tensor-core harness. Three cuBLAS baselines (FP32, TF32,
// BF16), then each registered TF32/BF16 kernel is verified against the
// SAME-precision cuBLAS result and timed. Separate binary from bench.cu so
// the FP32 ladder stays untouched. Usage:  ./bench_tc [M] [N] [K] [iters]
#include "common.cuh"
#include "gemm.h"
#include "gemm_tc.h"

// defined in v0_cublas.cu
float run_cublas(cublasHandle_t, int, int, int, float, float, const float*,
                 const float*, float*, int, int);
// defined in v0_cublas_tc.cu
float run_cublas_tf32(cublasHandle_t, int, int, int, float, float,
                      const float*, const float*, float*, int, int);
float run_cublas_bf16(cublasHandle_t, int, int, int, float, float,
                      const __nv_bfloat16*, const __nv_bfloat16*, float*, int,
                      int);

static float time_launcher(GemmLauncher fn, int M, int N, int K, float alpha,
                           float beta, const float* dA, const float* dB,
                           float* dC, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn(M, N, K, alpha, beta, dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t; t.start();
  for (int i = 0; i < iters; ++i) fn(M, N, K, alpha, beta, dA, dB, dC);
  return t.stop_ms() / iters;
}

static float time_launcher(BfLauncher fn, int M, int N, int K, float alpha,
                           float beta, const __nv_bfloat16* dA,
                           const __nv_bfloat16* dB, float* dC, int warmup,
                           int iters) {
  for (int i = 0; i < warmup; ++i) fn(M, N, K, alpha, beta, dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t; t.start();
  for (int i = 0; i < iters; ++i) fn(M, N, K, alpha, beta, dA, dB, dC);
  return t.stop_ms() / iters;
}

static void print_row(const char* name, const char* prec, const char* ms,
                      const char* gflops, const char* sol, const char* xfp32,
                      const char* verify, const char* errfp32) {
  printf("%-20s %-5s %10s %10s %8s %8s %18s %12s\n", name, prec, ms, gflops,
         sol, xfp32, verify, errfp32);
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 2048;
  int N = argc > 2 ? atoi(argv[2]) : 2048;
  int K = argc > 3 ? atoi(argv[3]) : 2048;
  int iters  = argc > 4 ? atoi(argv[4]) : 50;
  int warmup = 10;
  float alpha = 1.f, beta = 0.f;

  printf("GEMM-TC  M=%d N=%d K=%d  iters=%d\n", M, N, K, iters);
  printf("precisions: fp32 (storage=compute=FP32), "
         "tf32 (storage=FP32 pre-rounded, compute=TF32), "
         "bf16 (storage=compute=BF16, accumulate=FP32)\n");

  // ---- host inputs ------------------------------------------------------
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  fill_random(hA, 1); fill_random(hB, 2);

  std::vector<float> hA_tf32 = hA, hB_tf32 = hB;
  for (auto& x : hA_tf32) x = round_to_tf32(x);
  for (auto& x : hB_tf32) x = round_to_tf32(x);

  std::vector<__nv_bfloat16> hA_bf16 = to_bf16(hA);
  std::vector<__nv_bfloat16> hB_bf16 = to_bf16(hB);

  std::vector<float> h_out((size_t)M * N);
  std::vector<float> h_ref_fp32((size_t)M * N);
  std::vector<float> h_ref_tf32((size_t)M * N);
  std::vector<float> h_ref_bf16((size_t)M * N);

  // ---- device buffers -----------------------------------------------------
  float *dA, *dB, *dA_tf32, *dB_tf32, *dC;
  __nv_bfloat16 *dA_bf16, *dB_bf16;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dA_tf32, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB_tf32, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dA_bf16, hA.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dB_bf16, hB.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&dC, h_out.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_tf32, hA_tf32.data(), hA_tf32.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB_tf32, hB_tf32.data(), hB_tf32.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dA_bf16, hA_bf16.data(), hA_bf16.size()*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB_bf16, hB_bf16.data(), hB_bf16.size()*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));

  char buf_ms[32], buf_g[32], buf_sol[32], buf_x[32], buf_v[48], buf_e[32];
  print_row("version", "prec", "ms", "GFLOP/s", "%SoL", "xFP32", "verify", "errFP32");

  // ---- cuBLAS baselines, one per precision --------------------------------
  CUDA_CHECK(cudaMemset(dC, 0, h_out.size()*sizeof(float)));
  float ms_fp32 = run_cublas(handle, M, N, K, alpha, beta, dA, dB, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(h_ref_fp32.data(), dC, h_ref_fp32.size()*sizeof(float), cudaMemcpyDeviceToHost));
  double sol_fp32 = gemm_gflops(M, N, K, ms_fp32);
  snprintf(buf_ms, sizeof(buf_ms), "%.3f", ms_fp32);
  snprintf(buf_g, sizeof(buf_g), "%.1f", sol_fp32);
  print_row("cublas_fp32(SoL)", "fp32", buf_ms, buf_g, "100.0", "1.00", "-", "-");

  CUDA_CHECK(cudaMemset(dC, 0, h_out.size()*sizeof(float)));
  float ms_tf32 = run_cublas_tf32(handle, M, N, K, alpha, beta, dA_tf32, dB_tf32, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(h_ref_tf32.data(), dC, h_ref_tf32.size()*sizeof(float), cudaMemcpyDeviceToHost));
  double sol_tf32 = gemm_gflops(M, N, K, ms_tf32);
  double err_tf32_fp32 = max_rel_diff(h_ref_tf32, h_ref_fp32);
  snprintf(buf_ms, sizeof(buf_ms), "%.3f", ms_tf32);
  snprintf(buf_g, sizeof(buf_g), "%.1f", sol_tf32);
  snprintf(buf_x, sizeof(buf_x), "%.2f", ms_fp32 / ms_tf32);
  snprintf(buf_e, sizeof(buf_e), "%.2e", err_tf32_fp32);
  print_row("cublas_tf32(SoL)", "tf32", buf_ms, buf_g, "100.0", buf_x, "-", buf_e);

  CUDA_CHECK(cudaMemset(dC, 0, h_out.size()*sizeof(float)));
  float ms_bf16 = run_cublas_bf16(handle, M, N, K, alpha, beta, dA_bf16, dB_bf16, dC, warmup, iters);
  CUDA_CHECK(cudaMemcpy(h_ref_bf16.data(), dC, h_ref_bf16.size()*sizeof(float), cudaMemcpyDeviceToHost));
  double sol_bf16 = gemm_gflops(M, N, K, ms_bf16);
  double err_bf16_fp32 = max_rel_diff(h_ref_bf16, h_ref_fp32);
  snprintf(buf_ms, sizeof(buf_ms), "%.3f", ms_bf16);
  snprintf(buf_g, sizeof(buf_g), "%.1f", sol_bf16);
  snprintf(buf_x, sizeof(buf_x), "%.2f", ms_fp32 / ms_bf16);
  snprintf(buf_e, sizeof(buf_e), "%.2e", err_bf16_fp32);
  print_row("cublas_bf16(SoL)", "bf16", buf_ms, buf_g, "100.0", buf_x, "-", buf_e);

  // ---- narrow pass/fail tolerance: provisional, see CUDA_GEMM_v5계획.md
  // "v5-0" — the same-precision cuBLAS vs cuBLAS maxdiff above and the FP32
  // ladder's kernel-vs-cuBLAS maxdiff set the final value after this
  // harness's first real measurement.
  constexpr double kRelTol = 1e-3;
  int failures = 0;

  // ---- TF32 kernels, fed the pre-rounded TF32 inputs -----------------------
  {
    int n = 0; const TcVersion* tbl = tf32_table(n);
    for (int i = 0; i < n; ++i) {
      CUDA_CHECK(cudaMemset(dC, 0, h_out.size()*sizeof(float)));
      bool ok = tbl[i].fn(M, N, K, alpha, beta, dA_tf32, dB_tf32, dC);
      if (!ok) { print_row(tbl[i].name, "tf32", "-", "-", "-", "-", "skip", "-"); continue; }
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_out.data(), dC, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost));
      double diff_same = max_rel_diff(h_out, h_ref_tf32);
      double diff_fp32 = max_rel_diff(h_out, h_ref_fp32);
      bool pass = diff_same < kRelTol;
      if (!pass) ++failures;
      float ms = time_launcher(tbl[i].fn, M, N, K, alpha, beta, dA_tf32, dB_tf32, dC, warmup, iters);
      double g = gemm_gflops(M, N, K, ms);
      snprintf(buf_ms, sizeof(buf_ms), "%.3f", ms);
      snprintf(buf_g, sizeof(buf_g), "%.1f", g);
      snprintf(buf_sol, sizeof(buf_sol), "%.1f", 100.0 * g / sol_tf32);
      snprintf(buf_x, sizeof(buf_x), "%.2f", ms_fp32 / ms);
      snprintf(buf_v, sizeof(buf_v), "maxdiff=%.2e %s", diff_same, pass ? "PASS" : "FAIL");
      snprintf(buf_e, sizeof(buf_e), "%.2e", diff_fp32);
      print_row(tbl[i].name, "tf32", buf_ms, buf_g, buf_sol, buf_x, buf_v, buf_e);
    }
  }

  // ---- BF16 kernels, fed the BF16-converted inputs -------------------------
  {
    int n = 0; const BfVersion* tbl = bf16_table(n);
    for (int i = 0; i < n; ++i) {
      CUDA_CHECK(cudaMemset(dC, 0, h_out.size()*sizeof(float)));
      bool ok = tbl[i].fn(M, N, K, alpha, beta, dA_bf16, dB_bf16, dC);
      if (!ok) { print_row(tbl[i].name, "bf16", "-", "-", "-", "-", "skip", "-"); continue; }
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_out.data(), dC, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost));
      double diff_same = max_rel_diff(h_out, h_ref_bf16);
      double diff_fp32 = max_rel_diff(h_out, h_ref_fp32);
      bool pass = diff_same < kRelTol;
      if (!pass) ++failures;
      float ms = time_launcher(tbl[i].fn, M, N, K, alpha, beta, dA_bf16, dB_bf16, dC, warmup, iters);
      double g = gemm_gflops(M, N, K, ms);
      snprintf(buf_ms, sizeof(buf_ms), "%.3f", ms);
      snprintf(buf_g, sizeof(buf_g), "%.1f", g);
      snprintf(buf_sol, sizeof(buf_sol), "%.1f", 100.0 * g / sol_bf16);
      snprintf(buf_x, sizeof(buf_x), "%.2f", ms_fp32 / ms);
      snprintf(buf_v, sizeof(buf_v), "maxdiff=%.2e %s", diff_same, pass ? "PASS" : "FAIL");
      snprintf(buf_e, sizeof(buf_e), "%.2e", diff_fp32);
      print_row(tbl[i].name, "bf16", buf_ms, buf_g, buf_sol, buf_x, buf_v, buf_e);
    }
  }

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dA_tf32)); CUDA_CHECK(cudaFree(dB_tf32));
  CUDA_CHECK(cudaFree(dA_bf16)); CUDA_CHECK(cudaFree(dB_bf16));
  return failures ? 1 : 0;
}
