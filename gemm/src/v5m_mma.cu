// v5m: WMMA replaced by PTX ldmatrix + mma.sync (BF16 mma.sync.m16n8k16; TF32 mma.sync.m16n8k8, 2 per BK=16 step).
// Load paths per precision: A [BM][BK+pad] K-major via non-transposed ldmatrix for both; B [BK][BN+8] MN-major via ldmatrix.trans for BF16, plain LDS for TF32.
// Compared against the best same-precision WMMA kernel.
// STUB: not implemented, launchers return false so bench_tc prints skip.

#include <cuda_bf16.h>
#include "gemm_tc.h"

bool launch_v5m_tf32(int, int, int, float, float, const float*, const float*, float*) {
  return false;
}

bool launch_v5m_bf16(int, int, int, float, float, const __nv_bfloat16*, const __nv_bfloat16*, float*) {
  return false;
}
