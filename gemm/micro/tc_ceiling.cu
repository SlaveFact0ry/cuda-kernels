// tc_ceiling.cu — tensor-pipe utilization ceiling microbenchmark.
//
// Hypothesis under test: on GA102, NCU's
// sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
// caps at ~50% when the accumulator is FP32, because 100% on that metric
// corresponds to the FP16-accumulate issue rate, and FP32 accumulate halves
// it (TF32 quarters it). cuBLAS TF32/BF16 kernels showed ~49% while stalled
// on the math-pipe-throttle stall reason, consistent with "the pipe really
// is at its ceiling for this accumulate type", not "underutilized".
//
// Each kernel below is a closed loop of independent mma.sync.aligned PTX
// instructions with NO memory traffic: register fragments are initialized
// once from threadIdx (nonzero, lane-varying), then NUM_ACC independent
// accumulator chains run INNER_UNROLL*outer_iters back-to-back mma ops
// each. Independent chains hide the D-feeds-C latency of any single chain;
// the final store is guarded by a condition the device compiler cannot
// prove false at compile time (outer_iters is a runtime argument), so the
// whole computation must be retained even though that branch is never
// taken at the sizes we launch with.
//
// Variants (distinct __global__ kernels, selected by name on the CLI):
//   f16f16   mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16   (FP16 acc)
//   f16f32   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32   (FP32 acc)
//   bf16f32  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (FP32 acc)
//   tf32f32  mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32  (FP32 acc)
//
// Usage: ./tc_ceiling [variant]
//   no arg      -> run all four, print the summary table (plain benchmark)
//   variant arg -> one warmup launch + one timed launch of just that kernel
//                  (this is the shape `make profile-micro-tc` profiles with
//                  ncu --launch-skip 1 -c 1, so the profiled launch is the
//                  timed one, not the warmup)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "common.cuh"

// ---- register-fragment packing helpers -------------------------------------
// tf32 fragments are plain 32-bit registers holding an FP32 bit pattern with
// the low 13 mantissa bits cleared (same convention as gemm_tc.h's
// round_to_tf32, used by the TF32 kernel ladder) -- cvt.rna.tf32.f32's
// output layout. f16/bf16 fragments pack two elements per 32-bit register.
__device__ __forceinline__ uint32_t pack_half2(float x0, float x1) {
  __half2 h = __floats2half2_rn(x0, x1);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_bf16x2(float x0, float x1) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t pack_tf32(float x) {
  return __float_as_uint(x) & 0xffffe000u;
}

constexpr int kNumAcc       = 8;    // independent accumulator chains / warp
constexpr int kInnerUnroll  = 16;   // statically unrolled mma calls / chain / outer iter

// FLOPs for one mma.sync.m16n8kK op: 2 * M * N * K (mul+add per inner term).
constexpr double kFlopsF16K16  = 2.0 * 16 * 8 * 16;  // variants a/b/c
constexpr double kFlopsTf32K8  = 2.0 * 16 * 8 * 8;   // variant d

// ---- (a) f16 in, f16 accumulate --------------------------------------------
__global__ void mma_f16_f16acc_kernel(float* out, int outer_iters) {
  int lane = threadIdx.x & 31;
  float base = 1.0f + 0.01f * lane;  // nonzero, lane-varying

  uint32_t a0 = pack_half2(base + 0.10f, base + 0.20f);
  uint32_t a1 = pack_half2(base + 0.30f, base + 0.40f);
  uint32_t a2 = pack_half2(base + 0.50f, base + 0.60f);
  uint32_t a3 = pack_half2(base + 0.70f, base + 0.80f);
  uint32_t b0 = pack_half2(base + 0.11f, base + 0.21f);
  uint32_t b1 = pack_half2(base + 0.31f, base + 0.41f);

  uint32_t d0[kNumAcc], d1[kNumAcc];
#pragma unroll
  for (int k = 0; k < kNumAcc; ++k) {
    d0[k] = pack_half2(0.01f * (k + 1), 0.02f * (k + 1));
    d1[k] = pack_half2(0.03f * (k + 1), 0.04f * (k + 1));
  }

  for (int i = 0; i < outer_iters; ++i) {
#pragma unroll
    for (int j = 0; j < kInnerUnroll; ++j) {
#pragma unroll
      for (int k = 0; k < kNumAcc; ++k) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
            : "+r"(d0[k]), "+r"(d1[k])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }

  if (outer_iters == -1) {  // never true at runtime; keeps the loop live
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * kNumAcc * 2;
#pragma unroll
    for (int k = 0; k < kNumAcc; ++k) {
      out[idx + 2 * k]     = __uint_as_float(d0[k]);
      out[idx + 2 * k + 1] = __uint_as_float(d1[k]);
    }
  }
}

// ---- (b) f16 in, f32 accumulate --------------------------------------------
__global__ void mma_f16_f32acc_kernel(float* out, int outer_iters) {
  int lane = threadIdx.x & 31;
  float base = 1.0f + 0.01f * lane;

  uint32_t a0 = pack_half2(base + 0.10f, base + 0.20f);
  uint32_t a1 = pack_half2(base + 0.30f, base + 0.40f);
  uint32_t a2 = pack_half2(base + 0.50f, base + 0.60f);
  uint32_t a3 = pack_half2(base + 0.70f, base + 0.80f);
  uint32_t b0 = pack_half2(base + 0.11f, base + 0.21f);
  uint32_t b1 = pack_half2(base + 0.31f, base + 0.41f);

  float d0[kNumAcc], d1[kNumAcc], d2[kNumAcc], d3[kNumAcc];
#pragma unroll
  for (int k = 0; k < kNumAcc; ++k) {
    d0[k] = 0.01f * (k + 1); d1[k] = 0.02f * (k + 1);
    d2[k] = 0.03f * (k + 1); d3[k] = 0.04f * (k + 1);
  }

  for (int i = 0; i < outer_iters; ++i) {
#pragma unroll
    for (int j = 0; j < kInnerUnroll; ++j) {
#pragma unroll
      for (int k = 0; k < kNumAcc; ++k) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0[k]), "+f"(d1[k]), "+f"(d2[k]), "+f"(d3[k])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }

  if (outer_iters == -1) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * kNumAcc * 4;
#pragma unroll
    for (int k = 0; k < kNumAcc; ++k) {
      out[idx + 4*k]     = d0[k]; out[idx + 4*k + 1] = d1[k];
      out[idx + 4*k + 2] = d2[k]; out[idx + 4*k + 3] = d3[k];
    }
  }
}

// ---- (c) bf16 in, f32 accumulate -------------------------------------------
__global__ void mma_bf16_f32acc_kernel(float* out, int outer_iters) {
  int lane = threadIdx.x & 31;
  float base = 1.0f + 0.01f * lane;

  uint32_t a0 = pack_bf16x2(base + 0.10f, base + 0.20f);
  uint32_t a1 = pack_bf16x2(base + 0.30f, base + 0.40f);
  uint32_t a2 = pack_bf16x2(base + 0.50f, base + 0.60f);
  uint32_t a3 = pack_bf16x2(base + 0.70f, base + 0.80f);
  uint32_t b0 = pack_bf16x2(base + 0.11f, base + 0.21f);
  uint32_t b1 = pack_bf16x2(base + 0.31f, base + 0.41f);

  float d0[kNumAcc], d1[kNumAcc], d2[kNumAcc], d3[kNumAcc];
#pragma unroll
  for (int k = 0; k < kNumAcc; ++k) {
    d0[k] = 0.01f * (k + 1); d1[k] = 0.02f * (k + 1);
    d2[k] = 0.03f * (k + 1); d3[k] = 0.04f * (k + 1);
  }

  for (int i = 0; i < outer_iters; ++i) {
#pragma unroll
    for (int j = 0; j < kInnerUnroll; ++j) {
#pragma unroll
      for (int k = 0; k < kNumAcc; ++k) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0[k]), "+f"(d1[k]), "+f"(d2[k]), "+f"(d3[k])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }

  if (outer_iters == -1) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * kNumAcc * 4;
#pragma unroll
    for (int k = 0; k < kNumAcc; ++k) {
      out[idx + 4*k]     = d0[k]; out[idx + 4*k + 1] = d1[k];
      out[idx + 4*k + 2] = d2[k]; out[idx + 4*k + 3] = d3[k];
    }
  }
}

// ---- (d) tf32 in, f32 accumulate --------------------------------------------
__global__ void mma_tf32_f32acc_kernel(float* out, int outer_iters) {
  int lane = threadIdx.x & 31;
  float base = 1.0f + 0.01f * lane;

  // m16n8k8: A is m16k8 (4 regs, 1 elem/reg), B is n8k8 (2 regs, 1 elem/reg).
  uint32_t a0 = pack_tf32(base + 0.10f);
  uint32_t a1 = pack_tf32(base + 0.30f);
  uint32_t a2 = pack_tf32(base + 0.50f);
  uint32_t a3 = pack_tf32(base + 0.70f);
  uint32_t b0 = pack_tf32(base + 0.11f);
  uint32_t b1 = pack_tf32(base + 0.31f);

  float d0[kNumAcc], d1[kNumAcc], d2[kNumAcc], d3[kNumAcc];
#pragma unroll
  for (int k = 0; k < kNumAcc; ++k) {
    d0[k] = 0.01f * (k + 1); d1[k] = 0.02f * (k + 1);
    d2[k] = 0.03f * (k + 1); d3[k] = 0.04f * (k + 1);
  }

  for (int i = 0; i < outer_iters; ++i) {
#pragma unroll
    for (int j = 0; j < kInnerUnroll; ++j) {
#pragma unroll
      for (int k = 0; k < kNumAcc; ++k) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0[k]), "+f"(d1[k]), "+f"(d2[k]), "+f"(d3[k])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }

  if (outer_iters == -1) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * kNumAcc * 4;
#pragma unroll
    for (int k = 0; k < kNumAcc; ++k) {
      out[idx + 4*k]     = d0[k]; out[idx + 4*k + 1] = d1[k];
      out[idx + 4*k + 2] = d2[k]; out[idx + 4*k + 3] = d3[k];
    }
  }
}

// ---- host harness -----------------------------------------------------------
struct Variant {
  const char* name;
  void (*kernel)(float*, int);
  double flops_per_mma;
  double theoretical_tflops;  // at 1395 MHz, 82 SMs
};

static const Variant kVariants[] = {
  {"f16f16",  mma_f16_f16acc_kernel,  kFlopsF16K16, 117.1},
  {"f16f32",  mma_f16_f32acc_kernel,  kFlopsF16K16,  58.6},
  {"bf16f32", mma_bf16_f32acc_kernel, kFlopsF16K16,  58.6},
  {"tf32f32", mma_tf32_f32acc_kernel, kFlopsTf32K8,  29.3},
};
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

// 82 SMs * 4 blocks/SM * 256 threads/block, 8 warps/block.
constexpr int kBlocks        = 82 * 4;
constexpr int kThreadsPerBlk = 256;
constexpr int kOuterIters    = 20000;  // tuned for a multi-hundred-ms run

static void run_variant(const Variant& v, float* d_out, bool print_hdr) {
  // warmup
  v.kernel<<<kBlocks, kThreadsPerBlk>>>(d_out, kOuterIters);
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t; t.start();
  v.kernel<<<kBlocks, kThreadsPerBlk>>>(d_out, kOuterIters);
  float ms = t.stop_ms();

  long long total_warps = (long long)kBlocks * kThreadsPerBlk / 32;
  double mma_per_warp = (double)kOuterIters * kInnerUnroll * kNumAcc;
  double total_flops = (double)total_warps * mma_per_warp * v.flops_per_mma;
  double tflops = total_flops / (ms * 1e-3) / 1e12;
  double pct = 100.0 * tflops / v.theoretical_tflops;

  if (print_hdr) {
    printf("%-10s %10s %12s %10s\n", "variant", "ms", "TFLOP/s", "%theor.");
  }
  printf("%-10s %10.3f %12.2f %9.1f%%\n", v.name, ms, tflops, pct);
}

int main(int argc, char** argv) {
  size_t out_elems = (size_t)kBlocks * kThreadsPerBlk * kNumAcc * 4;
  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));

  if (argc > 1) {
    // single-variant mode: one warmup + one timed launch, for `ncu
    // --launch-skip 1 -c 1` to land exactly on the timed one.
    const Variant* v = nullptr;
    for (int i = 0; i < kNumVariants; ++i)
      if (strcmp(argv[1], kVariants[i].name) == 0) v = &kVariants[i];
    if (!v) {
      fprintf(stderr, "unknown variant '%s' (expected one of: ", argv[1]);
      for (int i = 0; i < kNumVariants; ++i) fprintf(stderr, "%s ", kVariants[i].name);
      fprintf(stderr, ")\n");
      return 1;
    }
    run_variant(*v, d_out, /*print_hdr=*/true);
  } else {
    printf("tc_ceiling  blocks=%d threads/blk=%d (warps/blk=%d)  "
           "outer_iters=%d inner_unroll=%d num_acc=%d\n",
           kBlocks, kThreadsPerBlk, kThreadsPerBlk / 32, kOuterIters,
           kInnerUnroll, kNumAcc);
    for (int i = 0; i < kNumVariants; ++i) run_variant(kVariants[i], d_out, i == 0);
  }

  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
