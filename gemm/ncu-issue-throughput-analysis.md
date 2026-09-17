# NCU Issue Throughput 분석 — Group A/B 클럭 민감도와 cp.async 일반화 가능성 검증

## 배경 및 동기

`gemm` 프로젝트(RTX 3090, sm_86)의 전 커널 버전을 4096³ 입력에 대해 두 조건 —
GPU boost clock, 그리고 `sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc 1395`로 고정한
base clock(1395MHz) — 각각에서 벤치마크한 결과, 커널들이 두 그룹으로 뚜렷이
나뉘는 것을 관찰했다.

- **Group A** (clock-robust): 클럭을 고정해 낮췄을 때 %SoL(Speed of Light)이
  거의 변하지 않거나 오히려 소폭 상승했다. `v3b_vectorize`
  (`register_tile_vec`), `v4b_doublebuffer_tp` (`double_buffer_transpose`),
  `v4c_warptiling` (`warptiling`), `v4d_wtdb` (`warptiling_doublebuffer`).
- **Group B** (clock-sensitive): 클럭을 고정해 낮췄을 때 %SoL이 눈에 띄게
  떨어졌다. `v4a_doublebuffer` (`double_buffer`), `v4_cp_async_pipe`
  (`cp_async_pipe`), `v4e_wtnt` (`warptiling_notranspose`).

이 A/B 분화의 1차 가설은 **issue throughput**(warp가 issue할 자격 있는
명령어를 갖지 못해 SM이 노는 현상)이 occupancy·DRAM 대역폭·bank conflict보다
더 직접적인 원인이라는 것이었다.

동시에, `docs/gemm-notes.md`의 `v4_cp_async_pipe` 항목에 이미 기록된 발견 —
cp.async(`__pipeline_memcpy_async`)의 16-byte 정렬 요구가 전치되지 않은
shared-memory destination(`As[BM][BK]`) 위에서 scalar fallback을 강제해
issue efficiency를 해친다는 관찰 — 을 "cp.async는 sm_86 SGEMM에서 레이아웃
제약 때문에 net loss다"라는 **일반화된 주장**으로 확장할 수 있는지를 검증하는
것이 이번 조사의 두 번째 목적이었다. Group B에 cp.async를 쓰지 않는
`double_buffer`, `warptiling_notranspose`도 포함되어 있다는 점이 이 일반화
주장에 대한 자연스러운 반증 후보였다.

## 방법론

1. 락클럭 상태 확인: `nvidia-smi --query-gpu=persistence_mode,clocks.current.sm --format=csv`
   결과 persistence mode enabled, 1395MHz로 고정된 것을 확인했다.
2. `/home/soondol/cuda-kernels/gemm`에서 `make clean && make bench` 실행.
   Makefile의 `NVFLAGS`에는 `-Xptxas -v`가 조건 없이 항상 포함되어 있어,
   빌드 시 커널별 register/spill 카운트가 ptxas stderr로 출력된다 — 이를
   그대로 캡처했다(build log 경로는 아래 참고).
3. 위 7개 커널 함수 각각에 대해 `ncu`를 두 pass로 실행:
   - Pass 1 (Scheduler / Occupancy / Throughput):
     `ncu --set full -k <kernel_fn> -c 1 --page details --csv ./bench 4096 4096 4096`
   - Pass 2 (stall-reason breakdown, LDGSTS, shared LD/ST 명령어 수,
     bank-conflict 합계):
     `ncu -k <kernel_fn> -c 1 --page raw --csv --metrics <stall-reason 및
     shared-memory 명령어/bank-conflict 메트릭 목록> ./bench 4096 4096 4096`
   - 모든 실행은 `docs/gemm-notes.md`에 이미 기록된 기존 NCU 항목들과
     직접 비교 가능하도록 동일한 입력 shape(4096³, grid (32,32), 256
     threads/block)을 사용했다.
4. 재현성 교차검증: `register_tile_vec`의 `shared_ld.sum`(134,217,728),
   `double_buffer`의 `shared_ld.sum`(335,544,320),
   `cp_async_pipe`의 `shared_ld.sum`(348,102,656)/`LDGSTS`(8,388,608)를
   재측정한 결과, `docs/gemm-notes.md`에 기존에 기록된 값과 정확히 일치했다
   — 이번 측정 방법론이 기존 기록과 일관됨을 확인.
5. Raw CSV 출력물은 세션 스크래치 경로
   `/tmp/claude-1000/-home-soondol-cuda-kernels-gemm/0cb574ce-e24f-46ac-99a2-4460e1c410e7/scratchpad/ncu/ncu_<kernel>_details.csv`
   (occupancy/scheduler/throughput)와
   `/tmp/claude-1000/-home-soondol-cuda-kernels-gemm/0cb574ce-e24f-46ac-99a2-4460e1c410e7/scratchpad/ncu/ncu_<kernel>_metrics.csv`
   (stall ratio/LDGSTS/shared 명령어/bank conflict)에, 빌드 로그는
   `/tmp/claude-1000/-home-soondol-cuda-kernels-gemm/0cb574ce-e24f-46ac-99a2-4460e1c410e7/scratchpad/build.log`에
   보관되어 있다. **이 경로는 세션 스크래치 영역이며 저장소에 커밋된
   영구 아카이브가 아니다** — 향후 참조 시에는 위 `ncu` 명령을 그대로 재실행해
   재생성해야 한다.

## 실측 데이터

### 락클럭(1395MHz) 4096³ 벤치마크, 100 iters — %SoL 컨텍스트

```
v1_naive               76.229ms   1803.0 GFLOP/s    8.8% SoL
v2_smem_tiling         61.430ms   2237.3 GFLOP/s   10.9% SoL
v3_register_tile       18.307ms   7507.4 GFLOP/s   36.5% SoL
v3b_vectorize           8.700ms  15797.3 GFLOP/s   76.8% SoL
v4a_doublebuffer       11.512ms  11939.2 GFLOP/s   58.1% SoL
v4b_doublebuffer_tp     8.549ms  16077.1 GFLOP/s   78.2% SoL
v4_cp_async_pipe       10.790ms  12738.2 GFLOP/s   62.0% SoL
v4c_warptiling          7.882ms  17438.0 GFLOP/s   84.8% SoL
v4d_wtdb                7.843ms  17522.8 GFLOP/s   85.2% SoL
v4e_wtnt               10.630ms  12929.5 GFLOP/s   62.9% SoL
```

Group A(v3b, v4b, v4c, v4d)는 %SoL 76.8~85.2%로 상위권을 형성하고, Group B
(v4a, v4, v4e)는 58.1~62.9%로 뚜렷이 낮은 대역에 몰려 있다.

### Table 1 — Register/spill (7개 프로파일 커널)

| version | kernel fn | reg/thread | static smem | spill stores | spill loads |
|---|---|---|---|---|---|
| v3b_vectorize (A) | register_tile_vec | 113 | 8192B | 0 | 0 |
| v4b_doublebuffer_tp (A) | double_buffer_transpose | 127 | 16384B | 0 | 0 |
| v4c_warptiling (A) | warptiling | 99 | 8192B | 0 | 0 |
| v4d_wtdb (A) | warptiling_doublebuffer | 106 | 16384B | 0 | 0 |
| v4a_doublebuffer (B) | double_buffer | 102 | 16384B | 0 | 0 |
| v4_cp_async_pipe (B) | cp_async_pipe | 92 | 16384B | 0 | 0 |
| v4e_wtnt (B) | warptiling_notranspose | 128 | 8192B | 0 | 0 |

7개 커널 모두 spill이 0건이다. register는 92~128 reg/thread로 커널마다
다르지만, 전부 `Block Limit Registers=2` 문턱 안에 들어 있어 register
압박 자체가 occupancy 상한을 바꾸지 않는다(v3b~v4 계열에서 반복 확인된
기존 관찰과 동일). 즉 **register/spill 축은 A/B를 구분하지 못한다** — A와
B 양쪽에 92 reg부터 128 reg까지 뒤섞여 있다.

### Table 2 — Issue efficiency / occupancy / throughput

| kernel fn | Group | %SoL(locked) | Occ Theoretical/Achieved | Eligible Warps/Sched | No Eligible % | Warp Cycles/Issued Instr | Compute(SM)% | Memory% | DRAM% |
|---|---|---|---|---|---|---|---|---|---|
| register_tile_vec | A | 76.8 | 33.33/32.33 | 1.57 | 37.26 | 6.18 | 59.99 | 82.28 | 13.31 |
| double_buffer_transpose | A | 78.2 | 33.33/32.39 | 1.93 | 33.51 | 5.85 | 63.81 | 83.54 | 14.94 |
| warptiling | A | 84.8 | 33.33/32.29 | 1.91 | 31.34 | 5.64 | 65.95 | 61.03 | 14.81 |
| warptiling_doublebuffer | A | 85.2 | 33.33/32.26 | 1.73 | 29.92 | 5.53 | 67.05 | 61.11 | 14.60 |
| double_buffer | B | 58.1 | 33.33/32.38 | 1.45 | 44.63 | 7.02 | 54.33 | 91.88 | 10.22 |
| cp_async_pipe | B | 62.0 | 33.33/32.38 | 1.43 | 40.34 | 6.51 | 59.48 | 90.09 | 10.84 |
| warptiling_notranspose | B | 62.9 | 33.33/32.27 | 1.22 | 49.39 | 7.65 | 48.57 | 76.93 | 11.19 |

Achieved Occupancy(32.26~32.39%)와 DRAM Throughput(10.2~14.9%)은 7개 커널
전체에서 사실상 동일하다 — 이 두 축은 A/B 차이를 설명하지 못한다. 반면
**Eligible Warps/Scheduler, No Eligible %, Warp Cycles Per Issued
Instruction**은 A(1.57~1.93 / 29.9~37.3% / 5.53~6.18)와
B(1.22~1.45 / 40.3~49.4% / 6.51~7.65)로 **겹침 없이** 깨끗하게 갈린다.
Compute(SM)%도 A가 60~67%, B가 49~59%로 일관되게 A가 높다 —
"issue할 자격 있는 warp가 없어 SM이 논다"는 가설과 정확히 부합하는
패턴이다.

### Table 3 — Shared-memory 명령어 수 / LDGSTS / bank conflict (raw sums, 단일 4096³ 실행)

| kernel fn | Group | LDS(shared_ld).sum | STS(shared_st).sum | LDGSTS.sum | bank-conflict(LD).sum | bank-conflict(ST).sum | top-2 stall reasons |
|---|---|---|---|---|---|---|---|
| register_tile_vec | A | 134,217,728 | 20,971,520 | 0 | 268,435,456 | 16,777,216 | not_selected 31.3% / short_scoreboard 24.6% |
| double_buffer_transpose | A | 134,217,728 | 20,971,520 | 0 | 268,435,456 | 16,777,216 | not_selected 43.0% / mio_throttle 21.3% |
| warptiling | A | 134,217,728 | 20,971,520 | 0 | 0 | 16,777,216 | not_selected 41.6% / long_scoreboard 21.2% |
| warptiling_doublebuffer | A | 134,217,728 | 20,971,520 | 0 | 0 | 16,777,216 | short_scoreboard 37.0% / not_selected 34.6% |
| double_buffer | B | 335,544,320 | 8,388,608 | 0 | 536,870,912 | 0 | mio_throttle 39.2% / not_selected 27.6% |
| cp_async_pipe | B | 348,102,656 | 0 | 8,388,608 | 536,870,912 | 0 | mio_throttle 30.4% / not_selected 26.0% |
| warptiling_notranspose | B | 134,217,728 | 8,388,608 | 0 | 402,653,184 | 0 | not_selected 24.2% / barrier 22.0% |

내부 일관성 체크: stall-reason 크기의 상대적 크기(A는 top-2 합산 비중이
대략 55~72% 범위, B는 56~70% 범위로 겹치긴 하나, 절대 카운트 기준으로는
A가 대략 1.24M~1.40M, B가 대략 1.69M~2.00M 수준 — 상대적 순위로서 Table 2의
Warp Cycles/Issued Instr 랭킹과 같은 방향을 가리킨다)는 Table 2의 결론을
뒷받침한다. 단 이 stall magnitude 비교는 절대 카운트가 아니라 순위
정합성으로 읽어야 한다.

## 판정 / 결론

### Verdict 1 — A/B 격차는 issue throughput으로 설명되는가: **YES**

Occupancy(32.26~32.39%, 전 커널 사실상 동일)와 DRAM Throughput(10.2~14.9%,
전 커널 사실상 동일)은 통계적으로 평평해 두 원인 모두 배제된다. 반면
Eligible Warps/Scheduler, No Eligible %, Warp Cycles Per Issued
Instruction은 A와 B 사이에 **겹침 없는** 분리를 보인다. Compute(SM)%도
A(60~67%)가 B(49~59%)보다 일관되게 높다 — 이는 "SM이 issue할 자격 있는
warp를 찾지 못해 논다"는 상태와 정확히 부합한다. 따라서 이번 A/B 클럭
민감도 분화를 가장 직접적으로 설명하는 층위는 **issue throughput**이며,
occupancy와 DRAM 대역폭은 (전 커널에서 사실상 동일하므로) 원인에서
배제된다.

다만 bank conflict는 occupancy·DRAM대역폭과 같은 층위의 배제 대상이
아니다. Bank conflict(4-way) → LDS wavefront 직렬화 → warp 간 도착
시각 편차 → barrier stall 증가 → eligible warp 감소 → issue throughput
저하로 이어지는 **상류(upstream) 원인** 중 하나이며, 이 verdict가
측정하는 issue throughput 지표들은 그 인과 사슬의 하류 증상 층위다.
실제로 `warptiling_notranspose`(v4e)는 bank-conflict(LD).sum이
402,653,184건(전치된 형제 커널 v4c/v4d는 0건)이면서 동시에 barrier
stall이 22.0%(Group A 범위 7.0~15.2%)로 튀는 것이 함께 관찰되며, 이
경로는 Verdict 2-(2)에서 v4e의 issue starvation 원인으로 별도
설명한다.

### Verdict 2 — "cp.async는 sm_86 SGEMM에서 레이아웃 제약 때문에 net loss다"는 일반화 가능한 주장인가: **NO — v4_cp_async_pipe 개별 사례만 설명하며 일반화되지 않는다**

Group B 내부에는 뿌리가 다른 **두 가지 독립적인 원인**이 섞여 있다.

**(1) Write-side(staging) 레이아웃 문제 — `double_buffer`(v4a)와
`cp_async_pipe`(v4)에 공통.**
두 커널 모두 shared_ld.sum이 Group A 기준선(134M)의 약 2.5배(335M/348M)이며,
`mio_throttle`이 stall reason 1위다(39.2%/30.4% — Group A의 어떤 커널에서도
`mio_throttle`이 1위인 경우는 없다). 원인은 non-transposed `As[BM][BK]`
레이아웃이 `regM` 벡터화를 깨뜨려 scalar LDS로 쪼개지는 것이다 — 이 문제는
cp.async 도입 이전(v4a, cp.async 미사용)부터 이미 존재했다.
`cp_async_pipe`는 여기에 LDGSTS 8,388,608건을 추가하고 shared_ld도
v4a보다 오히려 더 많다(348M > 335M)는 점에서, cp.async가 이 write-side
병목을 **만들지도 고치지도** 않았다는 것이 확인된다. 다만 stall 구성은
다소 이동한다 — `mio_throttle`이 39.2%→30.4%로 줄고 short_scoreboard가
14.9%→23.1%로 늘어나는데, 이는 프로젝트에 기존에 문서화된 "MIO-throttle
stall ratio 2.29→1.64" 발견(`docs/gemm-notes.md` v4_cp_async_pipe 항목)과
방향이 일치하며 이번 독립 측정으로 다시 확인됐다. Warp Cycles/Issued
Instr도 7.02→6.51로 소폭 개선된다. 즉 cp.async는 write-side 병목 위에
얹힌 **중립적인 2차 요인**이다 — 진짜 병목(레이아웃)을 건드리지 못한 채
issue 효율을 미세하게만 바꾼다.

**(2) Read-side(compute-loop access pattern) 레이아웃 문제 —
`warptiling_notranspose`(v4e) 단독.**
v4e의 shared_ld.sum은 134,217,728로 **Group A 기준선과 완전히 동일**하고
LDGSTS는 0이다 — cp.async도, v4a/v4식 write-side 벡터화 문제도 전혀
관여하지 않는다. 대신 bank-conflict(LD).sum이 402,653,184인데(전치된
형제 커널 v4c/v4d는 0), `barrier` stall이 22.0%로 Group A의 7.0~15.2%
범위를 크게 벗어난다. 이는 v4c/v4d가 사용하는 전치(transpose)를 생략한
데서 나오는, write-side와는 전혀 다른 **read-side** 레이아웃 문제다.

이 둘의 인과 관계는 다음 사슬로 이어진다: **bank conflict(4-way) → LDS
wavefront 직렬화 → warp 간 도착 시각 편차 → barrier stall 증가(22.0%)
→ eligible warp 감소(1.22, Group A 최저치보다도 낮음) → issue
throughput 저하**. 즉 bank conflict는 Verdict 1이 배제한 occupancy·
DRAM 대역폭과 달리 issue throughput과 경쟁하는 원인이 아니라, v4e에서
issue throughput 저하를 만들어내는 상류 메커니즘이다.

**최종 결론.** 이번 sm_86 SGEMM 스윕에서 issue-throughput 병목의 진짜
공통 원인은 **shared-memory 레이아웃 선택**(write/staging 측이든
read/compute-loop 측이든)이지, cp.async가 아니다. cp.async는 병목을
고치지도 악화시키지도 않는 중립적 요인이며, 조사한 Group B 3개 커널 중
`v4_cp_async_pipe` 단 하나에만 우연히 존재했을 뿐이다. 따라서
"cp.async는 sm_86 SGEMM에서 net loss다"라는 claim은 `v4_cp_async_pipe`
개별 사례를 넘어 인용 가능한 일반 명제로 확장할 수 없다 — 대신 "shared
memory layout이 issue throughput을 좌우하며, cp.async는 그 레이아웃
문제와 독립적이다"가 더 정확하고 인용 가능한 결론이다.

## 참고 (Raw data / 재현성)

- Raw CSV: `/tmp/claude-1000/-home-soondol-cuda-kernels-gemm/0cb574ce-e24f-46ac-99a2-4460e1c410e7/scratchpad/ncu/ncu_<kernel>_details.csv`,
  `ncu_<kernel>_metrics.csv`
- Build log(ptxas register/spill 출력 포함):
  `/tmp/claude-1000/-home-soondol-cuda-kernels-gemm/0cb574ce-e24f-46ac-99a2-4460e1c410e7/scratchpad/build.log`
- 위 경로는 세션 스크래치 영역이라 저장소에 커밋되지 않는다. 영구
  아카이브가 필요하면 "방법론" 절의 `ncu` 명령을 그대로 재실행해
  재생성할 것.
