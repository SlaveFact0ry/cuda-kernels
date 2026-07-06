// v4_pipelined.cu — YOUR RUNG. *This is the rung Ampere unlocks.*
//
// On the old Turing card this step was a "manual" double buffer because
// cp.async did not exist. On the 3090 (sm_80+) you have asynchronous
// global->shared copies, so v4 becomes a real SOFTWARE PIPELINE — the same
// idea CUTLASS uses.
//
// Two combinable ideas on top of v3:
//   (a) VECTORIZED transfers: move A/B tiles in 16-byte chunks (float4). This
//       maps perfectly onto cp.async's 16-byte mode.
//   (b) cp.async MULTI-STAGE PIPELINE: issue the copy for tile (k+1)[..k+S-1]
//       into separate shared-memory buffers while the MMA math for tile k runs,
//       then wait only on the stage you are about to consume. Hides global
//       latency behind compute.
//
// Primitives (#include <cuda_pipeline.h>):
//   __pipeline_memcpy_async(smem_ptr, gmem_ptr, 16);  // 16B async copy
//   __pipeline_commit();                              // close a stage
//   __pipeline_wait_prior(S-1);                       // wait until <=S-1 in flight
//   __syncthreads();                                  // before reading that smem
//
// Shape: keep v3's BM=128,BN=128,BK=8 micro-tiling, but allocate the A/B smem
// tiles S times (S = 2..4 stages) and round-robin the buffer index.
//
// Watch with NCU (this is the payoff measurement):
//   - "long scoreboard" / memory stalls should DROP vs v3 (latency now hidden);
//   - DRAM throughput as % of peak (~936 GB/s on the 3090);
//   - did issue efficiency / achieved occupancy hold while latency hid?
//   - Ampere note: cp.async.cg bypasses L1 and lands in shared via L2; reason
//     about whether that helps your reuse pattern.
//
// This rung is the single most "modern GPU" thing in the repo — make the
// before/after stall breakdown the centerpiece of docs/notes.md.
#include "gemm.h"

bool launch_v4_vector(int, int, int, float, float, const float*, const float*, float*) {
  return false;  // not implemented yet -> harness skips it
}
