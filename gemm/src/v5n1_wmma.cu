// v5n1: naive WMMA GEMM (one warp per 16x16 C tile, operands read straight from global).
// C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N], row-major, FP32 accumulate.
// Precision is a template parameter so the _tf32 and _bf16 builds differ only in Traits.
//
// Assumptions (same as the FP32 ladder): M, N multiples of 32, K multiple of 16.
// Build: nvcc -arch=sm_86 -O3 -lineinfo ...   (bf16/tf32 WMMA need sm_80+)
// Wired into bench_tc via launch_v5n1_tf32 / launch_v5n1_bf16 (include/gemm_tc.h).

#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "common.cuh"   // CUDA_CHECK
#include "gemm_tc.h"    // BfLauncher signature, GemmLauncher

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Precision traits
// ---------------------------------------------------------------------------
struct Bf16Traits {
    using Elem  = __nv_bfloat16;          // storage type of A/B in global memory
    using FragT = __nv_bfloat16;          // fragment element type
    static constexpr int MMA_K = 16;      // m16n16k16
    template <class Frag>
    __device__ static void convert(Frag&) {}   // nothing to do
};

struct Tf32Traits {
    using Elem  = float;                  // TF32 is stored as FP32
    using FragT = wmma::precision::tf32;
    static constexpr int MMA_K = 8;       // m16n16k8
    template <class Frag>
    __device__ static void convert(Frag& f) {
        // Element-to-thread mapping is unspecified, but applying the same op to
        // every element is safe. Inputs are pre-rounded on the host, so this is
        // an identity op (toggle it off for H6: results must be bit-identical).
#pragma unroll
        for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
    }
};

constexpr int WMMA_M = 16, WMMA_N = 16;
constexpr int BK = 16;                    // k per outer step (fixed for both precisions)
constexpr int WARPS_M = 2, WARPS_N = 2;   // 4 warps per block -> block covers 32x32 of C

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <class Tr>
__global__ void v5n1_wmma(const typename Tr::Elem* __restrict__ A,
                          const typename Tr::Elem* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K, float alpha, float beta) {
    const int warp_id = threadIdx.x / 32;
    const int tile_m  = (blockIdx.y * WARPS_M + warp_id / WARPS_N) * WMMA_M;  // row of C tile
    const int tile_n  = (blockIdx.x * WARPS_N + warp_id % WARPS_N) * WMMA_N;  // col of C tile

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, Tr::MMA_K, typename Tr::FragT, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, Tr::MMA_K, typename Tr::FragT, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, Tr::MMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll                             // keep unroll explicit (lesson from the FP32 ladder)
        for (int kk = 0; kk < BK; kk += Tr::MMA_K) {
            // ldm is in elements: A is row-major with row stride K, B with row stride N.
            wmma::load_matrix_sync(a, A + (size_t)tile_m * K + (k0 + kk), K);
            wmma::load_matrix_sync(b, B + (size_t)(k0 + kk) * N + tile_n, N);
            Tr::convert(a);
            Tr::convert(b);
            wmma::mma_sync(acc, a, b, acc);
        }
    }
    // C = alpha*acc + beta*C. Same-shape accumulator fragments share the
    // element-to-thread mapping, so combining them element-wise is safe.
    float* c_ptr = C + (size_t)tile_m * N + tile_n;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, Tr::MMA_K, float> c_old;
    wmma::load_matrix_sync(c_old, c_ptr, N, wmma::mem_row_major);
#pragma unroll
    for (int i = 0; i < acc.num_elements; ++i) acc.x[i] = alpha * acc.x[i] + beta * c_old.x[i];
    wmma::store_matrix_sync(c_ptr, acc, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
template <class Tr>
static bool launch_v5n1(int M, int N, int K, float alpha, float beta,
                        const typename Tr::Elem* A, const typename Tr::Elem* B, float* C) {
    constexpr int BM = WMMA_M * WARPS_M, BN = WMMA_N * WARPS_N;
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr,
            "v5n1: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
        return false;                        // harness skips the row
    }
    dim3 block(32 * WARPS_M * WARPS_N);
    dim3 grid(N / BN, M / BM);               // exact by the guard above
    v5n1_wmma<Tr><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool launch_v5n1_tf32(int M, int N, int K, float alpha, float beta,
                      const float* dA, const float* dB, float* dC) {
    return launch_v5n1<Tf32Traits>(M, N, K, alpha, beta, dA, dB, dC);
}

bool launch_v5n1_bf16(int M, int N, int K, float alpha, float beta,
                      const __nv_bfloat16* dA, const __nv_bfloat16* dB, float* dC) {
    return launch_v5n1<Bf16Traits>(M, N, K, alpha, beta, dA, dB, dC);
}
