// v2_smem_tiling.cu — YOUR RUNG.
//
// Goal: cut global-memory traffic by staging BM x BK and BK x BN tiles of A and
// B into __shared__ memory, so each loaded element is reused by a whole block
// instead of by one thread.
//
// Suggested starting shape (square tiles):
//   #define BM 32   // rows of C per block
//   #define BN 32   // cols of C per block
//   #define BK 32   // contraction-dim tile
//   block = 32x32 threads, one thread -> one C element (still).
//
// Sketch:
//   __shared__ float As[BM][BK], Bs[BK][BN];
//   float acc = 0;
//   for (int t = 0; t < K; t += BK) {
//       // cooperatively load A[.., t..t+BK] and B[t..t+BK, ..] into As/Bs
//       __syncthreads();
//       for (int k = 0; k < BK; ++k) acc += As[ty][k] * Bs[k][tx];
//       __syncthreads();
//   }
//   write alpha*acc + beta*C.
//
// Things to verify with NCU after it works:
//   - global memory transactions drop sharply vs v1 (the whole point);
//   - check for shared-memory BANK CONFLICTS on the Bs[k][tx] / As[ty][k] reads;
//   - are global loads now COALESCED during the tile load? (P&H ch.5 territory).
//
// Ampere note: sm_86 lets you opt into larger shared memory per block (up to
// ~100 KB vs 64 KB on Turing) via
//   cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
// so you can afford bigger tiles (more reuse) here than the old card allowed.
//
// Log the before/after numbers in docs/notes.md — that log becomes your paper.
#include "gemm.h"

bool launch_v2_smem(int, int, int, float, float, const float*, const float*, float*) {
  return false;  // not implemented yet -> harness skips it
}
