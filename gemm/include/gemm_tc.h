#pragma once
#include <cmath>
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
// Tables list the tensor-core kernels (v5n1 implemented, v5n2..v5s, v5m stubs); the cuBLAS baselines live
// in bench_tc.cu. To add a row, declare its launcher below and add
// {"name", launcher} to the matching table.
using BfLauncher = bool (*)(int M, int N, int K, float alpha, float beta,
                            const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                            float* dC);

bool launch_v5n1_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5n1_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
bool launch_v5n2_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5n2_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
bool launch_v5n3_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5n3_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
bool launch_v5n4_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5n4_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
bool launch_v5n5_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5n5_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
bool launch_v5n6_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5n6_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
bool launch_v5s_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5s_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);
bool launch_v5m_tf32(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5m_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*);

struct TcVersion { const char* name; GemmLauncher fn; };
struct BfVersion { const char* name; BfLauncher fn; };

inline const TcVersion* tf32_table(int& count) {
  static const std::vector<TcVersion> table = {
    {"v5n1_tf32", launch_v5n1_tf32},
    {"v5n2_tf32", launch_v5n2_tf32},
    {"v5n3_tf32", launch_v5n3_tf32},
    {"v5n4_tf32", launch_v5n4_tf32},
    {"v5n5_tf32", launch_v5n5_tf32},
    {"v5n6_tf32", launch_v5n6_tf32},
    {"v5s_tf32", launch_v5s_tf32},
    {"v5m_tf32", launch_v5m_tf32},
  };
  count = (int)table.size();
  return table.data();
}

inline const BfVersion* bf16_table(int& count) {
  static const std::vector<BfVersion> table = {
    {"v5n1_bf16", launch_v5n1_bf16},
    {"v5n2_bf16", launch_v5n2_bf16},
    {"v5n3_bf16", launch_v5n3_bf16},
    {"v5n4_bf16", launch_v5n4_bf16},
    {"v5n5_bf16", launch_v5n5_bf16},
    {"v5n6_bf16", launch_v5n6_bf16},
    {"v5s_bf16", launch_v5s_bf16},
    {"v5m_bf16", launch_v5m_bf16},
  };
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

// Normwise relative error ||out - ref||_F / ||ref||_F (double accumulation).
// Unlike common.cuh's max_rel_diff (denominator floored at 1, so it reads as
// an ABSOLUTE error on the many C elements with |C| < 1 when inputs are
// uniform in [-1,1]), this metric reflects the actual precision loss of the
// reduced-precision compute path and stays roughly size-independent as K
// grows, since both norms grow with K at the same rate.
inline double normwise_rel_diff(const std::vector<float>& out,
                                const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    double d = (double)out[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return std::sqrt(num) / std::sqrt(den);
}
