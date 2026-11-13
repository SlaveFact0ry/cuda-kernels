// v5n4: 128-bit global->smem loads + 16 B smem row padding (ldm).
// FP32 counterpart: v3b + v4f.
// STUB: not implemented, launchers return false so bench_tc prints skip.

#include <cuda_bf16.h>
#include "gemm_tc.h"

bool launch_v5n4_tf32(int, int, int, float, float, const float*, const float*, float*) {
  return false;
}

bool launch_v5n4_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*) {
  return false;
}
