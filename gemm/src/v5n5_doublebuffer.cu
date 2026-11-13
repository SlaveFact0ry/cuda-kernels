// v5n5: double buffer via registers.
// FP32 counterpart: v4a.
// STUB: not implemented, launchers return false so bench_tc prints skip.

#include <cuda_bf16.h>
#include "gemm_tc.h"

bool launch_v5n5_tf32(int, int, int, float, float, const float*, const float*, float*) {
  return false;
}

bool launch_v5n5_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*) {
  return false;
}
