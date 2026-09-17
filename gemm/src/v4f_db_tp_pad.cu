
#include "common.cuh"
#include "gemm.h"

#define BN 128
#define BM 128
#define BK 8
#define TM 8
#define TN 8

// NOTE: 원 사양은 BANK_PAD=1(float 1개)였으나, pad(c)=c+(c>>5)*1은 c=32,64,96에서
// 각각 +1,+2,+3이 되어 4-float(float4=16B) 정렬이 깨지고 misaligned address로 크래시한다
// (직접 빌드해서 확인함). float4 벡터 로드/스토어를 유지하려면 패딩 단위가 4의 배수여야
// 하므로 BANK_PAD를 float4 1개(4 floats) 단위로 교정한다 — 32-원소 세그먼트마다
// float4 1개(16B)를 패딩. pad(c)는 c가 항상 4의 배수이므로 이 값도 항상 4의 배수로
// 유지되어 벡터화 로드/스토어가 안전하다.
#define BANK_PAD 4
__device__ __forceinline__ int pad(int c) { return c + (c >> 5) * BANK_PAD; }

__global__ void double_buffer_transpose_pad(int M, int N, int K, float alpha, float beta,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C){

    // 적재 좌표 계산은 v3b와 동일(float4 폭 기준으로 tid를 접는다).
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int innerRowA = tid / (BK/4); int innerColA = tid % (BK/4);   // A: 128행 x float4 2개
    int innerRowB = tid / (BN/4); int innerColB = tid % (BN/4);   // B: 8행 x float4 32개
    int col0      = innerColA * 4;
    int col0B     = innerColB * 4;

    __shared__ float As[2][BK][BM], Bs[2][BK][BN + (BN/32)*BANK_PAD];
    float pf_a[4], pf_b[4];
    float regM[TM], regN[TN];
    float acc[TM][TN] = {0.f};
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // ---- prologue: tile 0 -> buffer 0 ----
    *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0]);
    *reinterpret_cast<float4*>(&pf_b) = *reinterpret_cast<const float4*>(&B[innerRowB * N + col0B ]);
    As[0][col0][innerRowA]= pf_a[0];
    As[0][col0+1][innerRowA] = pf_a[1];
    As[0][col0+2][innerRowA] = pf_a[2];
    As[0][col0+3][innerRowA] = pf_a[3];
    *reinterpret_cast<float4*>(&Bs[0][innerRowB][pad(col0B)]) = *reinterpret_cast<const float4*>(&pf_b);
    __syncthreads();
    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        int nxt = t + BK;
        if (nxt < K) {
            *reinterpret_cast<float4*>(&pf_a) = *reinterpret_cast<const float4*>(&A[innerRowA* K +col0 + nxt]);
            *reinterpret_cast<float4*>(&pf_b) = *reinterpret_cast<const float4*>(&B[(innerRowB+nxt) * N  + col0B]);
        }
        // (2) compute on CURRENT buffer — the ENTIRE k-loop
        // 계산은 cur 버퍼만 건드린다. 위에서 날린 로드는 cur^1로 갈 것이므로 충돌 없음.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            *reinterpret_cast<float4*>(&regM[0]) = *reinterpret_cast<float4*>(&As[cur][k][threadIdx.y*TM]);
            *reinterpret_cast<float4*>(&regM[4]) = *reinterpret_cast<float4*>(&As[cur][k][threadIdx.y*TM + 4]);
            *reinterpret_cast<float4*>(&regN[0]) = *reinterpret_cast<float4*>(&Bs[cur][k][pad(threadIdx.x*TN)]);
            *reinterpret_cast<float4*>(&regN[4]) = *reinterpret_cast<float4*>(&Bs[cur][k][pad(threadIdx.x*TN+4)]);
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        // (3) store prefetched tile into the OTHER buffer — ONE sync, THEN swap
        if (nxt < K) {
            As[cur^1][col0][innerRowA] = pf_a[0];
            As[cur^1][col0+1][innerRowA] = pf_a[1];
            As[cur^1][col0+2][innerRowA] = pf_a[2];
            As[cur^1][col0+3][innerRowA] = pf_a[3];
            *reinterpret_cast<float4*>(&Bs[cur^1][innerRowB][pad(col0B)]) = *reinterpret_cast<const float4*>(&pf_b);
            __syncthreads();
            cur ^= 1;                                        // 버퍼 교체
        }

    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int r = threadIdx.y*TM + i, c = threadIdx.x*TN + j;
            C[r*N + c] = alpha*acc[i][j] + beta*C[r*N + c];
        }
}

bool launch_v4f_db_tp_pad(int M, int N, int K, float alpha, float beta,
                    const float* dA, const float* dB, float* dC) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr,
            "v4f_db_tp_pad: requires M,N,K multiples of %d/%d/%d (got %d,%d,%d)\n",
            BM, BN, BK, M, N, K);
    return false;                    // harness skips the row
    }
    dim3 block(BN / TN, BM / TM);              // x -> N (col), y -> M (row)
    dim3 grid(div_up(N, BN), div_up(M, BM));   // exact by the guard above
    double_buffer_transpose_pad<<<grid, block>>>(M, N, K, alpha, beta, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

