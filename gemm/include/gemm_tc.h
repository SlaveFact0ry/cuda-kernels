#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_bf16.h>
#include "gemm.h"  // GemmLauncher

// Tensor-core launcher tables, parallel to gemm.h's GemmLauncher/Version but
// split by precision:
//   - TF32 kernels reuse GemmLauncher: inputs stay FP32 (TF32 is a compute
//     mode, not a storage type), fed pre-rounded FP32 data (round_to_tf32).
//   - BF16 kernels get their own signature: inputs are __nv_bfloat16*.
//
// Both tables are empty for now (v5-0 ships no kernels yet, only the cuBLAS
// baselines in bench_tc.cu). Add rows as v5n1_tf32, v5n1_bf16, ... land:
//   bool launch_v5n1_tf32(int, int, int, float, float, const float*, const float*, float*);
//   bool launch_v5n1_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
// then add {"v5n1_tf32", launch_v5n1_tf32} / {"v5n1_bf16", launch_v5n1_bf16}
// to the tables below.
using BfLauncher = bool (*)(int M, int N, int K, float alpha, float beta,
                            const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                            float* dC);

struct TcVersion { const char* name; GemmLauncher fn; };
struct BfVersion { const char* name; BfLauncher fn; };

inline const TcVersion* tf32_table(int& count) {
  static const std::vector<TcVersion> table = {};
  count = (int)table.size();
  return table.data();
}

inline const BfVersion* bf16_table(int& count) {
  static const std::vector<BfVersion> table = {};
  count = (int)table.size();
  return table.data();
}

// ---- host precision helpers -------------------------------------------------

// Round FP32 to the nearest TF32-representable value (ties away from zero),
// matching PTX cvt.rna.tf32.f32: keep 10 explicit mantissa bits, drop 13.
// Pre-rounding the host inputs means the kernel's own TF32 conversion
// (identical bit pattern in, identical bit pattern out) becomes a no-op, so
// cuBLAS and our kernels see exactly the same values regardless of either
// side's internal rounding mode.
inline float round_to_tf32(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  if ((u & 0x7f800000u) != 0x7f800000u) {  // leave inf/nan untouched
    u += 0x00001000u;                      // half of the dropped range
    u &= 0xffffe000u;                      // clear the low 13 bits
  }
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

// Host-side FP32 -> BF16 conversion (round-to-nearest-even, via
// __float2bfloat16), done once outside the timed region so the kernel and
// cuBLAS both read the same BF16 bytes from global memory.
inline std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& v) {
  std::vector<__nv_bfloat16> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16(v[i]);
  return out;
}
