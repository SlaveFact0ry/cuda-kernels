# GEMM 커널 NCU 메트릭 비교

측정 조건: `ncu -k <kernel_fn> -c 1 --page raw --csv --metrics <...> ./bench 4096 4096 4096` (RTX 3090, sm_86, boost clock — 락클럭 아님). 총 11개 구현 커널 (`v5_wmma_tc`는 미구현 stub이라 제외).

## 1. Shared Memory 뱅크 충돌 (Bank Conflict)

`v4a` vs `v4b(tp)` vs `v4f(swizzle)` vs `v4e(nt)` 비교용.

| version | kernel fn | bank_conflicts LD (sum) | bank_conflicts ST (sum) |
|---|---|---:|---:|
| v1_naive | `naive_sgemm` | 0 | 0 |
| v2_smem_tiling | `smem_sgemm` | 0 | 30,458,322 |
| v3_register_tile | `register_tile` | 536,870,912 | 0 |
| v3b_vectorize | `register_tile_vec` | 268,435,456 | 16,777,216 |
| v4a_doublebuffer | `double_buffer` | 536,870,912 | 0 |
| v4b_doublebuffer_tp | `double_buffer_transpose` | 268,435,456 | 16,777,216 |
| v4f_db_tp_swizzle | `double_buffer_transpose_swizzle` | 0 | 16,777,216 |
| v4_cp_async_pipe | `cp_async_pipe` | 536,870,912 | 0 |
| v4c_warptiling | `warptiling` | 0 | 16,777,216 |
| v4d_wtdb | `warptiling_doublebuffer` | 0 | 16,777,216 |
| v4e_wtnt | `warptiling_notranspose` | 402,653,184 | 0 |

## 2. 글로벌 메모리 트래픽 및 벡터화 검증 (Memory & Vectorization)

`v2`, `v3` tiling과 `v3b` vectorization 적용 시 변화 확인용.

| version | kernel fn | dram_bytes_read (sum) | dram_bytes_write (sum) | inst_executed_pipe_lsu (sum) |
|---|---|---:|---:|---:|
| v1_naive | `naive_sgemm` | 17,627,059,584 | 69,465,088 | 4,297,064,448 |
| v2_smem_tiling | `smem_sgemm` | 8,828,984,320 | 69,391,744 | 2,954,887,168 |
| v3_register_tile | `register_tile` | 619,570,304 | 68,761,088 | 403,734,528 |
| v3b_vectorize | `register_tile_vec` | 989,166,208 | 66,655,488 | 164,659,200 |
| v4a_doublebuffer | `double_buffer` | 1,004,225,152 | 66,646,400 | 357,605,376 |
| v4b_doublebuffer_tp | `double_buffer_transpose` | 1,016,177,280 | 66,816,512 | 164,659,200 |
| v4f_db_tp_swizzle | `double_buffer_transpose_swizzle` | 1,176,863,104 | 66,665,216 | 164,659,200 |
| v4_cp_async_pipe | `cp_async_pipe` | 1,004,143,616 | 66,852,992 | 365,953,024 |
| v4c_warptiling | `warptiling` | 993,425,536 | 66,888,320 | 164,642,816 |
| v4d_wtdb | `warptiling_doublebuffer` | 980,376,448 | 66,804,096 | 164,659,200 |
| v4e_wtnt | `warptiling_notranspose` | 1,017,968,384 | 66,749,312 | 152,059,904 |

## 3. 스톨(대기) 비율 분석 (Latency Hiding)

Double Buffering(`v4a`), `cp.async` 파이프라인의 효과 확인용. 단위 %.

| version | kernel fn | long_scoreboard stall % | short_scoreboard stall % | math_pipe_throttle stall % |
|---|---|---:|---:|---:|
| v1_naive | `naive_sgemm` | 22.58 | 0.01 | 0.09 |
| v2_smem_tiling | `smem_sgemm` | 14.12 | 0.41 | 0.47 |
| v3_register_tile | `register_tile` | 21.15 | 41.31 | 0.39 |
| v3b_vectorize | `register_tile_vec` | 13.94 | 19.18 | 0.09 |
| v4a_doublebuffer | `double_buffer` | 0.41 | 12.39 | 0.35 |
| v4b_doublebuffer_tp | `double_buffer_transpose` | 0.59 | 10.55 | 1.04 |
| v4f_db_tp_swizzle | `double_buffer_transpose_swizzle` | 0.55 | 7.31 | 1.51 |
| v4_cp_async_pipe | `cp_async_pipe` | 0.91 | 19.08 | 0.46 |
| v4c_warptiling | `warptiling` | 15.98 | 9.32 | 0.11 |
| v4d_wtdb | `warptiling_doublebuffer` | 12.18 | 28.54 | 0.11 |
| v4e_wtnt | `warptiling_notranspose` | 12.56 | 10.24 | 0.08 |

> 참고: 사용자가 지정한 메트릭명 `smsp__warp_issue_stalled_math_pipe_per_warp_active.pct`는 이 ncu 버전(2024.1.1.0)에 존재하지 않아 요청 시 조용히 드롭됨(에러 없음, 컬럼 자체가 누락). `ncu --query-metrics`로 확인한 올바른 이름 `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct`로 대체 측정함.

## 4. 하드웨어 활용률 (Throughput)

메모리 바운드 vs 컴퓨트 바운드 판별용 (%SoL과 직결). 단위 % of peak sustained elapsed.

| version | kernel fn | sm__throughput.avg % | dram__throughput.avg % | l1tex__throughput.avg % |
|---|---|---:|---:|---:|
| v1_naive | `naive_sgemm` | 98.57 | 25.37 | 98.57 |
| v2_smem_tiling | `smem_sgemm` | 84.20 | 15.91 | 84.20 |
| v3_register_tile | `register_tile` | 38.86 | 4.12 | 57.93 |
| v3b_vectorize | `register_tile_vec` | 60.08 | 13.33 | 82.39 |
| v4a_doublebuffer | `double_buffer` | 54.32 | 10.21 | 91.87 |
| v4b_doublebuffer_tp | `double_buffer_transpose` | 63.82 | 13.91 | 83.56 |
| v4f_db_tp_swizzle | `double_buffer_transpose_swizzle` | 71.60 | 17.77 | 62.40 |
| v4_cp_async_pipe | `cp_async_pipe` | 59.50 | 10.92 | 90.11 |
| v4c_warptiling | `warptiling` | 65.94 | 14.77 | 61.03 |
| v4d_wtdb | `warptiling_doublebuffer` | 67.04 | 14.65 | 61.11 |
| v4e_wtnt | `warptiling_notranspose` | 48.57 | 11.19 | 76.92 |

