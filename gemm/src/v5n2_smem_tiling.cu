// v5n2: block tile 64x64x16 in smem, 16 warps each one fragment.
// FP32 counterpart: v2.
// STUB: not implemented, launchers return false so bench_tc prints skip.

#include <cuda_bf16.h>
#include "gemm_tc.h"

bool launch_v5n2_tf32(int, int, int, float, float, const float*, const float*, float*) {
  return false;
}

bool launch_v5n2_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*) {
  return false;
}
