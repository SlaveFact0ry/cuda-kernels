// v5n3: block 128x128, 8 warps, warp tile 64x32 (4x2 fragments).
// FP32 counterpart: v3.
// STUB: not implemented, launchers return false so bench_tc prints skip.

#include <cuda_bf16.h>
#include "gemm_tc.h"

bool launch_v5n3_tf32(int, int, int, float, float, const float*, const float*, float*) {
  return false;
}

bool launch_v5n3_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*) {
  return false;
}
