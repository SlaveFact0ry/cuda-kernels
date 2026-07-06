// v3_register_tile.cu — YOUR RUNG.
//
// Goal: raise arithmetic intensity by having each THREAD compute a small TM x TN
// micro-tile of C (e.g. 8x8), holding partial sums in REGISTERS. This amortizes
// shared-memory reads across many FMAs and is usually the single biggest jump.
//
// Typical shape:
//   BM=128, BN=128, BK=8;  TM=8, TN=8;  block = (BM/TM)*(BN/TN) = 16x16 threads.
//   Each thread keeps float acc[TM][TN] in registers.
//   Inner loop: load a TM-slice of As and a TN-slice of Bs into registers, then
//   do TM*TN FMAs.
//
// Watch with NCU:
//   - occupancy vs ILP trade-off: bigger TM/TN -> more registers/thread -> lower
//     occupancy but more work per thread. Find the sweet spot empirically.
//     (This is exactly the pipelining/ILP intuition from P&H ch.4.)
//   - is it now COMPUTE-bound, or still stalled on shared-memory throughput?
#include "gemm.h"

bool launch_v3_regtile(int, int, int, float, float, const float*, const float*, float*) {
  return false;  // not implemented yet
}
