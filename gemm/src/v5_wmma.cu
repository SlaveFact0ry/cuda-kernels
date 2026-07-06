// v5_wmma_tc.cu — YOUR RUNG (stretch).
//
// Use the tensor cores via the WMMA API (#include <mma.h>,
// namespace nvcuda::wmma). Ampere (sm_86) gives you more options than Turing:
//
//   - fp16  (half)            16x16x16, fp32 accumulate   <- start here
//   - bf16  (__nv_bfloat16)   16x16x16, fp32 accumulate
//   - tf32  (precision::tf32) 16x16x8,  fp32 accumulate   <- near-fp32 range,
//                                                            tensor-core speed;
//                                                            great middle ground
//
//   wmma::fragment<matrix_a, 16,16,16, half, row_major> a_frag;
//   wmma::fragment<matrix_b, 16,16,16, half, row_major> b_frag;
//   wmma::fragment<accumulator,16,16,16, float>         c_frag;
//   wmma::fill_fragment(c_frag, 0.0f);
//   for (k tiles) { load_matrix_sync(...); mma_sync(c_frag,a_frag,b_frag,c_frag);}
//   store_matrix_sync(...);
//
// Verification: inputs are now reduced precision, so compare against a matching
// cuBLAS path (cublasGemmEx with CUDA_R_16F / CUDA_R_32F-TF32 compute) rather
// than the fp32 SoL — otherwise the diff you see is precision, not a bug. For
// tf32, set the cuBLAS math mode to CUBLAS_TF32_TENSOR_OP_MATH for an apples-to-
// apples ceiling.
//
// Point for the writeup: the tensor-core compute roof is far above fp32 (~142
// TFLOP/s fp16 on the 3090), so the bottleneck flips hard back to FEEDING the
// cores — smem bandwidth and your cp.async staging from v4 become the limiter.
// This closes the arc: every rung moved the bottleneck, and the last one moves
// it back to memory at a much higher absolute throughput. Best roofline story.
//
// Combine with v4: the strongest version is cp.async-fed WMMA. That is, in
// miniature, what cuBLAS/CUTLASS actually do.
#include "gemm.h"

bool launch_v5_wmma(int, int, int, float, float, const float*, const float*, float*) {
  return false;  // not implemented yet
}
