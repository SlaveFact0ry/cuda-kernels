# Optimization log

One entry per version. Write it the day you implement the rung, while the NCU
numbers are fresh — this log is the skeleton of the paper's results section.

Template:

## vN_name (date)
- **change:** what you did in one line.
- **result:** ms / GFLOP·s / %SoL at 2048³ (and other shapes if relevant).
- **bottleneck before:** what NCU said was limiting the previous version
  (e.g. DRAM throughput X% of peak, low arithmetic intensity, smem bank
  conflicts, occupancy Y%).
- **bottleneck after:** what limits THIS version now.
- **why it worked (or didn't):** the mechanism, tied to a concept
  (P&H ch.5 memory hierarchy / ch.4 ILP, PMPP tiling, etc).
- **surprises:** anything that didn't match your prediction.

---

## 참고 메모 (소스 주석에서 옮김)

- **NCU 체크리스트 (타일링 계열 커널 공통)**: 구현 후 확인할 것 —
  global memory transaction이 이전 버전 대비 확 줄었는지, shared-memory
  타일 로드 시 global load가 coalesced인지(P&H 5장), `As[ty][k]`/`Bs[k][tx]`
  같은 shared-memory 읽기에 bank conflict가 있는지.
- **Ampere 큰 smem note**: sm_86은 `cudaFuncSetAttribute(kernel,
  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`로 블록당 shared
  memory를 (Turing의 64KB 대비) 최대 ~100KB까지 opt-in할 수 있어서 더 큰
  타일(더 많은 재사용)을 쓸 수 있다. **v2에는 해당 없음** — v2는 static
  smem 8KB(`As[32][32]+Bs[32][32]`)로 opt-in 없이도 충분하다. 이 note가
  실제로 의미를 갖는 건 v4(cp.async 멀티스테이지 버퍼)·v5(WMMA) 처럼
  타일이 커지는 시점부터다.

---

## v1_naive (2026-07-06)
- change: 베이스라인, C 원소 하나당 스레드 하나, 재사용 없음. 매 스레드가 A의
  한 행과 B의 한 열 전체를 매번 global memory에서 직접 읽는다.
- result:
  - 2048³: 7.763 ms, 2213.1 GFLOP/s, 9.9% SoL (cuBLAS 22247.4 GFLOP/s)
  - 4096³: 63.888 ms, 2151.2 GFLOP/s, 9.4% SoL (cuBLAS 22873.3 GFLOP/s)
- bottleneck (NCU 실측 확인, 1024³, `make profile KERNEL=naive_sgemm`):
  **DRAM-bound가 아님** — DRAM Throughput이 peak의 1.64%밖에 안 돼서, 처음
  예측(memory-bound = DRAM bandwidth 문제)과 정반대다. 진짜 병목은
  **L1TEX/LSU 파이프**: Memory Throughput 97.56%, L1/TEX Cache Throughput
  97.97%, "Mem Pipes Busy" 97.56%. L1 Hit Rate는 87.5%로 데이터 대부분이
  L1에 남아있긴 하지만(A의 행은 half-warp 내에서 broadcast되고, B의 열 읽기는
  coalesced), 스레드마다 inner-loop 매 iteration마다 자기 몫의 L1TEX 요청을
  개별로 쏘기 때문에 그 요청 개수 자체가 DRAM bandwidth에 도달하기 훨씬 전에
  파이프를 포화시킨다. NCU가 uncoalesced global load를 직접 지적함(sector당
  32바이트 중 18바이트만 사용, 개선 여지 약 42.6%). Occupancy는 문제가 아님
  (95.1% 달성). Warp stall 분석: issue된 명령어 사이 40.0 사이클 중 65.9%가
  L1TEX/LG instruction queue 대기.
  - **계획 정정**: 프로파일링은 커널을 이름으로 지정해야 한다
    (`-k naive_sgemm` / `-k smem_sgemm`). `-k regex:".*" -c 1`은 프로세스
    전체에서 *가장 먼저 실행되는 커널*을 잡는데, `run_cublas()`가 버전
    루프보다 먼저 돌기 때문에 실제로는 cuBLAS 자체의 워밍업 커널
    (`ampere_sgemm_128x64_nn`)이 잡혔다. `gemm/Makefile`에
    `KERNEL ?= naive_sgemm` 변수를 추가해 수정함.
- evidence: [NCU details](ncu_naive_sgemm_details.csv)

---

## v2_smem_tiling (2026-07-11)
- change: A/B의 32×32 (`BM=BN=BK=32`) 타일을 블록마다 `__shared__` 메모리에
  올려서, 읽어온 원소를 스레드 하나가 아니라 타일 전체가 재사용하도록 함.
  아직 스레드 하나가 C 원소 하나를 계산(scalar accumulator, register
  tiling은 없음).
- result:
  - 2048³: 5.755 ms, 2985.1 GFLOP/s, 13.4% SoL (v1 대비 GFLOP/s +34.9%)
  - 4096³: 47.217 ms, 2910.8 GFLOP/s, 12.7% SoL (v1 대비 GFLOP/s +35.3%)
- bottleneck before: v1은 (위 정정 내용대로) DRAM-bound가 아니라 스레드별
  중복 global load로 인한 L1TEX/LSU 파이프 병목이었다.
- bottleneck after (NCU 실측 확인, 1024³, `make profile KERNEL=smem_sgemm`):
  병목이 L1TEX(global load)에서 **MIO 파이프(shared memory 명령어)**로
  옮겨감: Memory Throughput 84.06%, issue된 명령어 사이 36.9 사이클 중
  63.3%가 MIO instruction queue 대기 — 원인은 inner-loop마다 operand당
  scalar shared-memory load를 하나씩 쏘는 구조(`As[ty][k]`, `Bs[k][tx]`)로,
  v1과 같은 구조적 문제가 메모리 계층 한 단계 위로 옮겨간 것뿐이다. DRAM
  Throughput은 여전히 미미함(2.06%). Shared-store bank conflict는 부차적
  요인(shared-store wavefront의 24.0%가 conflict, 평균 1.3-way, 개선 여지
  약 21.1%). **Occupancy는 오히려 나빠짐**: 32×32=1024-thread 블록은 SM당
  블록 1개만 들어가서 theoretical occupancy가 66.7%(v1의 95.1%보다 낮음)로,
  레지스터·shared memory·블록당 warp 수 제한이 동시에 걸림. Issue Slots
  Busy도 v1보다 낮음(21.6% vs 28.6%).
- why it worked (or didn't): smem 타일링이 L1TEX 부하는 실제로 줄였고
  (Memory Throughput 97.6%→84.1%, GFLOP/s 향상과 일치) 방향 자체는 맞았다.
  하지만 "iteration당 scalar load 1개"라는 패턴을 없앤 게 아니라 shared
  memory로 옮겼을 뿐이라 MIO 파이프가 곧바로 새 한계로 등장했고, 여기에
  큰 블록 크기가 occupancy까지 깎아먹었다. P&H 5장 메모리 계층 관점으로
  보면: 타일링은 트래픽을 "어느 계층이 처리하느냐"만 바꿀 뿐, 스레드당
  ILP·재사용 여부가 그 새 계층이 병목이 될지를 결정한다.
- surprises: (1) 이 래더에서 v1은 어느 시점에도 DRAM-bound였던 적이 없다 —
  "memory-bound"라는 말이 실제로는 대역폭이 아니라 파이프/명령어 처리량
  문제였다는 점이 이 문제 크기에서는 교과서적 설명과 어긋난다. (2) v2의
  occupancy가 v1보다 나빠진 건 BM=BN=BK=32가 강제하는 1024-thread 블록의
  의도치 않은 대가로, v3에서 register tiling만 얹을 게 아니라 블록/타일
  크기 자체도 다시 검토할 가치가 있다. (3) v1·v2 모두 진짜 한계는
  명령어 레벨 문제다 — FMA 하나당 scalar memory op가 너무 많다. v3의
  register tiling(load 하나당 FMA를 늘리는 것)이 다음으로 맞는 방향이지,
  타일링을 더 하는 게 답이 아니다.
- evidence: [NCU details](ncu_smem_sgemm_details.csv) · [NCU summary](smem.csv) · [NCU screenshot](smem_sgemm.png)
- **bank-conflict discrepancy (unresolved, 2026-07-21)**: NCU's kernel-level
  rule reproduces this entry's number exactly -- 1.3-way avg conflict across
  24.32% of shared-store wavefronts (fresh `--set full` reprofile). But
  per-instruction source correlation (`ncu --page source`) reports
  `L1 Wavefronts Shared Excessive = 0` for every shared-memory instruction in
  the kernel, and static SASS tracing of the store address
  (`R6 = tx*4 + ty*128` → bank = tx, distinct per thread in a warp) predicts
  zero conflict for both `As[ty][tx]` and `Bs[ty][tx]`. Two NCU views from the
  same profiling run disagree, and the static analysis sides with "no
  conflict" -- root cause not identified (candidates: an NCU source-attribution
  limitation for STS, or a wavefront-accounting subtlety invisible in the
  source CSV). `v2_smem_tiling.cu`'s "bank-conflict-free" comment scopes
  itself to the inner-loop reads for this reason; don't extend that claim to
  the tile-load stores without resolving this first.

---

## v3_register_tile (2026-07-20)
- change: register/thread tiling — 스레드 하나가 C의 8×8(`TM=TN=8`) 마이크로
  타일 전체를 레지스터에 들고 계산한다(`BM=BN=128, BK=8`, 블록 = 16×16 =
  256 스레드). shared memory에서 읽은 값 하나를 v2처럼 FMA 1개에 쓰고
  버리는 게 아니라 `TM*TN=64`번의 FMA에 재사용해서, shared-memory read
  대비 연산 비율을 끌어올리는 게 목적이다.
- result:
  - 2048³: 1.954 ms, 8791.7 GFLOP/s, 40.7% SoL (cuBLAS 21579.8 GFLOP/s)
  - 4096³: 14.572 ms, 9431.4 GFLOP/s, 41.0% SoL (cuBLAS 23018.6 GFLOP/s)
- bottleneck before: v2는 scalar shared-memory load(`As[ty][k]`,
  `Bs[k][tx]`)가 iteration마다 하나씩 나가는 구조 때문에 MIO 파이프가
  포화(Mem Pipes Busy 84%)였고, 여기에 1024-thread 블록이 강제하는
  66.7% occupancy까지 겹쳐 있었다.
- bottleneck after (NCU 실측 확인, 1024³, `make profile KERNEL=register_tile`):
  병목의 성격이 v2의 MIO 파이프 throughput 포화(Mem Pipes Busy 84%)에서
  완전히 바뀌었다. v3는 register tiling으로 "operand 로드 1개당 FMA"
  비율을 8 shared-load → 64 FMA로 끌어올려서, Mem Pipes Busy가 20.0%로
  떨어졌다. 파이프 처리량 포화는 해소된 것이다. 그런데 compute-bound로
  넘어가지도 않았다: Compute(SM) Throughput 20.0%, FMA 파이프(fma-heavy
  10.3% / fma-lite 9.7%)로 연산 파이프는 여전히 한산하다. 실제 새 한계는
  register pressure가 만든 낮은 occupancy 때문에 memory latency를 숨길
  warp가 부족한 것이다. `acc[8][8]`(64개) + `regM[8]`+`regN[8]` 때문에
  스레드당 96 레지스터를 써서 Block Limit(Registers)=2로 theoretical
  occupancy가 33.3%(register-limited, scheduler당 4 warp, HW 최대 12의
  1/3)에 묶인다. 그 결과 scheduler당 active warp가 2.0개뿐이라 74.5%의
  사이클에 eligible warp가 하나도 없고(약 3.9 사이클마다 1회만 issue),
  warp가 issue 사이 7.84 사이클 중 short scoreboard(MIO/shared-mem 의존)
  2.68 사이클(34.2%) + long scoreboard(global-load 의존) 2.30 사이클
  (29.3%)로 memory latency 노출에 그대로 앉아 있다. 여기에
  `As[ty*TM+i][k]` 컬럼 접근이 shared load의 50.0% wavefront에서 평균
  3.2-way bank conflict(8.4M conflicts)를 일으켜 short-scoreboard latency를
  키운다(v2의 24% shared-store conflict보다 악화).
- why it worked (or didn't): 방향으로는 맞았다. v2 대비 GFLOP/s가 거의
  세 배로 뛰었다(2910.8 → 9431.4, +224%, 4096³ 기준). shared-memory
  transaction 하나당 arithmetic intensity를 끌어올려서 MIO 파이프를
  풀어준 게 그대로 처리량으로 이어졌다. 하지만 compute-bound까지는
  못 갔다: `TM=TN=8`이 요구하는 96 reg/thread 레지스터 풋프린트가
  occupancy를 낮게 눌러놔서, SM이 (빈도는 줄었지만 여전히 존재하는)
  memory latency를 숨길 만큼의 warp를 확보하지 못한다. P&H 4장/5장
  프레임으로 보면 v1·v2가 "명령어 처리량(파이프)" 문제였다면, v3는 같은
  ILP-vs-occupancy 트레이드오프가 레지스터 축으로 옮겨간 것이다. ILP를
  올리려고 쓴 레지스터가 occupancy를 깎아서 latency-hiding 여력을
  갉아먹는다.
- surprises: (1) NCU 프로파일은 1024³에서 떴는데(64 블록 = 8×8 grid vs
  82 SM → 0.39 wave, device underfill), 그래서 이 프로파일의 절대 SoL류
  퍼센트(Mem Throughput 30.3%, Compute 20.0%, DRAM 3.05%, L2 6.6%)는
  tail-effect로 눌려 있어 실제 4096³ 실행(41.1% SoL)을 그대로 대표하지
  않는다. 반면 per-warp 분석(register-capped occupancy, scoreboard stall
  breakdown, bank conflict)은 shape에 무관하므로 그대로 유효하다.
  Occupancy: Theoretical 33.33%(제한 자원 = 레지스터, 96 reg/thread,
  2 blocks/SM), Achieved 16.65%(7.99 warps/SM). achieved가 theoretical의
  절반인 건 0.39-wave underfill 때문이고, 4096³에서는 33%에 더 가깝게
  수렴할 것으로 예상된다. (2) bank conflict가 v2보다 오히려 악화됐다.
  v2는 shared-store wavefront의 24.0%에서 평균 1.3-way였는데, v3는
  shared-load wavefront의 50.0%에서 평균 3.2-way다. register blocking이
  도입한 `As[ty*TM+i][k]` 컬럼 접근 패턴 자체가 새로운 conflict 원인이라는
  뜻으로, v4/이후 작업에는 독립적인 레버가 두 개 남아 있다: occupancy를
  올리는 것(더 작은 TM/TN 또는 더 짧게 사는 레지스터)과, bank-conflict
  접근 패턴 자체를 고치는 것.

## v3b_vectorize (2026-08-17)
- change: v3의 shared-memory read를 벡터화했다. `As`를 `[BM][BK]`가 아니라
  전치된 `[BK][BM]` 레이아웃으로 스테이징하고, `regM`/`regN`을 채울 때
  v3처럼 `TM`/`TN`번의 scalar 원소 루프(`As[ty*TM+i][k]` 8회,
  `Bs[k][tx*TN+j]` 8회) 대신 `reinterpret_cast<float4*>`로 operand당
  2번의 float4 load로 8개 원소를 한 번에 읽는다. `BM=BN=128, BK=8,
  TM=TN=8`은 v3와 동일하게 유지.
- result (4096³, locked clock 기준— v0~v3와 동일 세션/조건):
  - 11.979 ms, 11473.1 GFLOP/s, 55.8% SoL (v0 cuBLAS 20543.9 GFLOP/s)
  - v3(7490.5 GFLOP/s) 대비 +53.2% GFLOP/s, 같은 locked-clock 실행 기준.
- bottleneck before: v3는 register tiling으로 MIO 파이프 포화는 풀었지만
  (Mem Pipes Busy 84%→20.0%, 1024³ 프로파일) compute-bound로 못 넘어갔다.
  96 reg/thread가 강제하는 33.3% theoretical occupancy(Block Limit
  Registers=2) 때문에 memory latency를 숨길 warp가 부족했고,
  `As[ty*TM+i][k]` 컬럼 접근이 shared-load wavefront의 50.0%에서 평균
  3.2-way bank conflict를 냈다.
- bottleneck after (NCU 실측 확인, 4096³, `register_tile_vec` 커널,
  `ncu --import v3b.ncu-rep`): **주의: 이 프로파일은 4096³(grid 1024
  블록/256 스레드)에서 떴고, 위 v3 항목의 NCU 수치(Mem Pipes Busy 20.0%,
  3.2-way/50% bank conflict)는 `make profile` 기본 타겟인 1024³에서 뜬
  것이다. 두 항목의 원시 퍼센트 대부분(Memory Throughput 67.19%,
  Compute(SM) 56.21%, DRAM 8.78% 등)은 shape이 달라 1:1로 직접 비교할 수
  없다.** 다만 레지스터 풋프린트/occupancy는 shape 무관이라 안전하게
  비교 가능한데: 96 reg/thread, Block Limit Registers=2, Theoretical
  Occupancy 33.33%(achieved 32.36%)로 v3와 완전히 동일하다 — read 쪽만
  벡터화해서는 occupancy가 전혀 바뀌지 않는다는 뜻이다.

  occupancy·conflict 구조는 그대로거나 오히려 나빠졌는데도 Compute(SM)
  Throughput은 56.21%까지 올랐고, NCU rule engine은 FMA를 "the
  highest-utilized pipeline (43.5%)... well-utilized, but should not be
  a bottleneck"이라고 지목한다(v3의 fma-heavy 10.3%/fma-lite 9.7%에서
  큰 도약). Scheduler는 One or More Eligible 59.36% / No Eligible
  40.64%, Active Warps Per Scheduler 3.89, Eligible Warps Per Scheduler
  1.51, Warp Cycles Per Issued Instruction 6.55(scheduler당 약 1.7
  사이클마다 1회 issue)로, v3보다 issue 여건이 개선됐다.

  bank conflict는 벡터화에도 불구하고 오히려 악화됐다: shared-load는
  134,217,728 요청 전체에서 평균 5.0-way conflict(268,435,456
  conflicts, 전체 671,098,541 wavefront의 40.00%) — v3의 3.2-way보다
  way-count는 늘고 wavefront 비율은 줄었다(40% vs 50%, shape 차이는
  감안해야 한다). 게다가 v3에서는 전혀 언급되지 않았던 **shared-store
  bank conflict**가 새로 나타난다: 33,554,432 store 요청에서 평균
  4.5-way(117,440,512 conflicts, 전체 150,995,251 wavefront의
  77.78%) — 전치된 `As[BK][BM]` 스테이징 레이아웃이 store address
  패턴을 바꿨을 가능성이 있지만, SASS 레벨 원인은 아직 추적하지
  않았다(v2 항목의 "unresolved discrepancy"와 같은 성격의 open item으로
  남겨둔다).

  Global load는 sector당 28.9/32바이트 사용(개선여지 약 6.5%), global
  store는 4.0/32바이트(개선여지 약 58.8%) — 둘 다 v3와 동일한 기존
  패턴(coalesced C-store 미적용)이고 v3b가 새로 만든 문제는 아니다.
  DRAM Throughput 8.78%로 여전히 DRAM-bound는 아니다.
- why it worked (or didn't): occupancy도 안 오르고 bank conflict도
  개선되지 않았는데(오히려 store conflict가 새로 생겼는데) GFLOP/s는
  v3 대비 +53.2% 올랐다. 데이터가 뒷받침하는 설명은 "conflict 해소"가
  아니라 "명령어당 처리량"이다: `TM`/`TN` 루프 8회의 scalar shared
  load를 float4 load 2회로 줄이면, 각 살아남은 로드 명령어가(bank
  conflict 때문에) 더 많이 replay되긴 해도, k-iteration 하나를 도는 데
  필요한 shared-load *명령어 개수* 자체가 크게 줄어든다. 그 결과
  issue된 명령어당 retire되는 FMA 비율이 올라갔고, 이게 그대로
  Compute(SM) Throughput 20.0%→56.21%, FMA pipe ~10%→43.5%로
  나타났다. occupancy(33.33%, 동일)나 bank conflict(오히려 악화)가
  고쳐져서 빨라진 게 아니라, 명령어 발행 자체가 줄어든 것이 병목을
  완화시켰다는 뜻이다.
- surprises: (1) 벡터화가 오히려 bank conflict를 키웠다 — load는
  3.2-way→5.0-way, 게다가 v3에서는 안 보이던 store conflict(4.5-way,
  wavefront의 77.78%)까지 새로 생겼다. 직관과 반대되는 결과라 SASS
  레벨 원인 추적이 필요하다(미확인, open item). (2) occupancy가 조금도
  개선되지 않았다 — read 벡터화는 레지스터 사용량(96 reg/thread)에
  전혀 영향을 주지 않아서 v3와 Block Limit Registers=2/33.33%가
  정확히 동일하다. (3) 그럼에도 성능이 크게 오른 건 순전히 "명령어
  개수 감소로 인한 FMA-per-issued-instruction 상승" 효과로 보인다 —
  occupancy나 conflict가 아니라 issue-throughput이 이번 병목 완화의
  진짜 메커니즘이라는 뜻이라, v4 이후에는 (a) bank conflict 자체를
  고치는 레버와 (b) occupancy를 올리는 레버가 여전히 독립적으로 남아
  있다.
- evidence: [NCU details](ncu_register_tile_vec_details.csv)

## v3b_vectorize — global→shared 벡터화 (2026-08-18)
- change: 기존 v3b(shared-memory read만 벡터화)에 이어, global→shared 쓰기
  (A/B 타일 스테이징)도 float4로 벡터화했다. A는 `float4`로 4개 원소를
  한 번에 읽어 전치 레이아웃 `As[BK][BM]`에 4번의 scalar 쓰기로
  흩뿌리고(전치 때문에 쓰기 자체는 벡터화 불가), B는 읽기·쓰기 모두
  `reinterpret_cast<float4*>`로 한 번에 처리한다. 이전 버전의
  `strideA`/`strideB` 기반 strided 루프(스레드당 여러 번 반복)를 없애고
  스레드당 float4 로드 1회로 타일 전체를 커버하도록 `innerRowA/B`,
  `innerColA/B` 계산을 재구성했다(`BM=BN=128, BK=8, TM=TN=8`은 변경 없음).
- result (4096³, locked clock 기준, v0~v3b와 동일 세션/조건):
  - 8.701 ms, 15795.3 GFLOP/s, 76.9% SoL (v0 cuBLAS 20535.1 GFLOP/s)
  - 이전 v3b(11473.1 GFLOP/s, 55.8% SoL) 대비 +37.7% GFLOP/s.
- bottleneck before: 이전 v3b는 shared-memory read만 벡터화해서
  occupancy(33.3%, 96 reg/thread)는 그대로 두고 명령어 개수만 줄여
  Compute(SM) 56.2%까지 올렸지만, global load는 여전히 strided scalar
  루프였고 shared-store bank conflict가 4.5-way(wavefront의 77.78%)로
  새로 나타난 상태였다(read 벡터화가 전치 레이아웃의 write 패턴을
  건드리지 않았기 때문).
- bottleneck after (NCU 실측, 4096³, `register_tile_vec` 커널,
  `ncu --import v3b2.ncu-rep`): 레지스터가 96→113 reg/thread로 늘었지만
  Block Limit Registers는 여전히 2, Theoretical Occupancy 33.33%(achieved
  32.33%)로 occupancy 상한 자체는 바뀌지 않았다(113 reg가 아직 128
  reg/thread 문턱을 넘지 않아 2 blocks/SM 유지). Compute(SM) Throughput은
  60.03%로 더 올랐고, FMA가 가장 바쁜 파이프(56.5%)로 확인된다. Scheduler
  One or More Eligible 62.75%(이전 59.36%), Warp Cycles Per Issued
  Instruction 6.18(이전 6.55, 즉 발행 간격이 더 짧아짐).

  Bank conflict는 read/write가 갈렸다: shared-load는 5.0-way/40.00%로
  이전과 완전히 동일하다(regM/regN을 채우는 `As[k][...]`/`Bs[k][...]`
  읽기 패턴 자체는 이번 변경에서 손대지 않았으므로 예상된 결과). 반면
  shared-store는 4.5-way(wavefront의 77.78%) → 2.4-way(wavefront의
  33.33%)로 뚜렷이 개선됐다 — B를 float4로 한 번에 쓰면서 store 요청
  수 자체도 33,554,432 → 20,971,520으로 줄었다.

  Global load 패턴은 28.9/32바이트로 이전과 동일하다(coalescing 개선
  여지 약 8%, 벡터화가 access pattern 자체를 바꾸진 않음 — sector
  활용률은 그대로). Global store는 4.0/32바이트(개선 여지 약 72%)로
  C store는 여전히 손대지 않은 기존 문제. DRAM Throughput은 13.54%
  (이전 8.78%)로 올랐지만 여전히 DRAM-bound는 아니다.
- why it worked (or didn't): occupancy는 그대로고 shared-load conflict도
  그대로인데 GFLOP/s가 37.7% 더 올랐다. v3b(read 벡터화)와 같은
  메커니즘의 연장선이다: strided scalar 루프로 여러 번 반복하던
  global→shared 로드를 스레드당 float4 로드 1회로 줄이면서, k-iteration을
  도는 데 필요한 로드 명령어 개수가 다시 줄었다. 여기에 B의 shared-store가
  float4로 합쳐지면서 store bank conflict까지 개선된 게 더해졌다(A는
  전치 레이아웃 때문에 여전히 4번의 scalar store지만, 최소한 global read는
  1번의 float4로 줄었다). 결과적으로 명령어당 처리량이 다시 올라
  Compute(SM) 56.2%→60.0%, FMA pipe 43.5%→56.5%로 이어졌다 — v3b
  항목에서 정리한 "occupancy·conflict가 아니라 issue-throughput이
  병목을 완화한다"는 결론이 이번에도 그대로 성립한다.
- surprises: (1) 레지스터가 96→113으로 17개나 늘었는데도(로컬 `float4
  tmp` + 인덱스 변수 증가 때문으로 추정) occupancy 상한(Block Limit
  Registers=2, 33.33%)은 전혀 바뀌지 않았다 — 128 reg/thread 문턱까지는
  아직 여유가 있다는 뜻이고, 뒤집어 말하면 register 예산이 15 reg/thread
  더 남아 있다는 뜻이기도 하다. (2) shared-store conflict가(완전히
  해소된 건 아니지만) 벡터화만으로 4.5-way→2.4-way, wavefront 비중
  77.78%→33.33%로 크게 줄었다 — v3b 항목에서 "SASS 레벨 원인 미확인"으로
  남겨뒀던 store conflict가 B의 벡터화 스토어만으로 상당 부분 완화된
  셈이지만, A는 전치 레이아웃 때문에 여전히 scalar store라 남은 open
  item(A store conflict의 SASS 레벨 원인)은 유효하다. (3) shared-load
  conflict(5.0-way)는 전혀 개선되지 않았다 — 이번 변경은 애초에 read
  경로를 건드리지 않았으므로 예상대로지만, v4 이후에도 "bank-conflict
  접근 패턴 자체를 고치는 레버"는 여전히 미해결로 남아 있다.
- evidence: [NCU details](ncu_register_tile_vec2_details.csv)

---

## v4a_doublebuffer (2026-08-21)
- change: 수동 소프트웨어 파이프라인(더블버퍼). `As`/`Bs`를 `[2][BM][BK]`/
  `[2][BK][BN]`로 두 벌 만들고, 매 k-타일 반복에서 (1) 다음 타일을
  레지스터(`pf_a`/`pf_b`)로 먼저 global load, (2) 현재 버퍼로 전체
  k-루프 compute, (3) prefetch해둔 값을 다른 버퍼에 store 후
  `__syncthreads()` 한 번으로 버퍼 swap(`cur ^= 1`) — global latency를
  compute 뒤에 숨기는 게 목표. `BM=BN=128, BK=8, TM=TN=8`은 이전과 동일.
- result (4096³, NCU 프로파일 세션 동일 실행 기준 — 이번 세션은 GPU
  클럭을 lock하지 않아 run-to-run 변동폭이 이전 항목들(≈locked)보다 약간
  크다, ±1~2%):
  - 11.529 ms, 11920.9 GFLOP/s, 58.7% SoL
  - v3b(15795.3 GFLOP/s, 76.9%) 대비 **-24.5% GFLOP/s — 역행(regression)**.
- bottleneck before: v3b는 113 reg/thread, Theoretical Occupancy 33.33%로
  register-limited, Compute(SM) 60.0%/FMA 56.5%까지 올라온 상태였다.
- bottleneck after (NCU 실측, 4096³, `double_buffer` 커널,
  `ncu_double_buffer_details.csv`): register는 102 reg/thread로 v3b보다
  오히려 줄었지만(`float4 pf_a/pf_b` 임시 배열이 늘었음에도) Block Limit
  Registers=2, Theoretical Occupancy 33.33%(achieved 32.38%)는 그대로다 —
  128 reg/thread 문턱 안에서는 register 수가 줄어도 occupancy 상한이 안
  바뀐다는 v3b 항목의 관찰이 여기서도 성립한다. Compute(SM) 54.32%, FMA
  42.6%로 v3b보다 **낮아졌다**(60.0%→54.32%, 56.5%→42.6%). Memory
  Throughput 91.86%, DRAM 10.33%. shared-load bank conflict는 3.2-way(v3b
  5.0-way보다 way-count는 줄었지만 wavefront 비중은 49.88%로 v3b의
  40.00%보다 늘었다). Scheduler는 1.8사이클당 1회 issue, active warp
  3.89, eligible warp 1.45/scheduler(v3b의 1.51보다 소폭 나쁨).

  결정적 지표는 shared-memory load 명령어 수다: `smsp__inst_executed_op_
  shared_ld.sum` = 335,544,320 — v3b의 134,217,728 대비 **2.5배**. 원인은
  레이아웃 변경: v3b는 `As`를 전치된 `[BK][BM]`로 둬서 `regM`을
  `reinterpret_cast<float4*>(&As[k][ty*TM])`로 2번의 float4 LDS에 8개
  원소를 담았다. v4a는 더블버퍼 async-copy 스테이징(다음 단계에서
  cp.async로 이어질 것을 염두에 둔 구조)을 위해 `As`를 `[BM][BK]`로
  바꿨는데, 그러면 `k`가 고정된 채 서로 다른 row `i`(=`ty*TM+i`)를 읽는
  것은 메모리상 BK=8칸씩 떨어진 strided 접근이 되어 float4로 묶을 수
  없다 — 그래서 `for (int i=0;i<TM;++i) regM[i]=As[cur][ty*TM+i][k]`처럼
  스칼라 LDS 8번으로 쪼개졌다.
- why it worked (or didn't): 안 됐다. 더블버퍼링이 숨기려 한 global
  latency보다, 레이아웃 전환이 만들어낸 shared-load 명령어 증가(2.5배)가
  더 크게 손해를 봤다. Occupancy는 그대로고 issue efficiency(eligible
  warp/scheduler)는 오히려 소폭 나빠졌다 — 알고리즘(더블버퍼 프리페치)
  자체는 맞는 방향이지만, 그걸 얹기 위해 바꾼 shared-memory 레이아웃이
  이전 rung(v3b)이 확보해둔 벡터화 read를 깨서 순손실이 났다.
- surprises: (1) register가 113→102로 줄었는데도 occupancy 상한은
  꼼짝하지 않았다 — v3b 항목의 "128 reg 문턱까지 15 reg 여유가 있다"는
  관찰이 여기서도 그대로 유효함을 재확인. (2) "prefetch로 latency
  숨기기"라는 원래 목표가 shared-load 명령어 수 폭증에 완전히 묻혔다 —
  이번 rung에서는 알고리즘 아이디어보다 레이아웃 선택이 실제 성능을 더
  크게 좌우했다.
- evidence: [NCU details](ncu_double_buffer_details.csv)

---

## v4_cp_async_pipe (2026-08-21)
- change: v4a와 동일한 `[2][BM][BK]`/`[2][BK][BN]` 더블버퍼 shape을 쓰되,
  global→shared 복사를 레지스터 경유 store 대신 Ampere의 cp.async
  (`__pipeline_memcpy_async`로 16B 비동기 복사, `__pipeline_commit`으로
  스테이지 봉인, `__pipeline_wait_prior(0)`로 대기)로 대체한 2-stage
  소프트웨어 파이프라인. `BM=BN=128, BK=8, TM=TN=8`은 동일.
- 버그 사냥 (구현 직후 `cudaDeviceSynchronize()` 단계에서 misaligned
  address 런타임 에러 발생, 3개를 순차로 발견/수정):
  1. steady-state에서 B의 다음 타일 prefetch가 stride로 `K`를 썼다
     (`&B[(innerRowB+nxt)*K + col0B]`) — B는 `K×N` 행렬이라 stride는
     `N`이어야 한다. `*K` → `*N`으로 수정.
  2. `innerRowA`/`innerColA`가 `tid / BK`, `tid % BK`로 계산돼 있었다
     (v3b/v4a는 `tid / (BK/4)`, `tid % (BK/4)`). 이 때문에 `innerRowA`가
     0..31만 커버해 BM=128행 중 1/4만 채워졌고, `col0(=innerColA*4)`이
     0..28까지 뻗어 BK=8 폭을 벗어났다.
  3. 4곳의 `__pipeline_memcpy_async` 목적지 주소가 전부 4-스케일된
     `col0`/`col0B`가 아니라 스케일 안 된 `innerColA`/`innerColB`를
     그대로 썼다 — 16B(4-float) 복사인데 목적지 오프셋이 4-float 배수가
     아닌 경우가 발생해 misaligned address를 유발했다. 4곳 모두
     `col0`/`col0B`로 수정.
- result (4096³, 같은 클럭-미고정 세션, ±1~2% 변동 감안):
  - 10.811 ms, 12712.4 GFLOP/s, 62.5% SoL
  - v4a(11920.9 GFLOP/s, 58.7%) 대비 +6.6% GFLOP/s, v3b(15795.3, 76.9%)
    대비는 여전히 -19.5% 낮음.
- bottleneck before: v4a는 102 reg/thread, 33.3% occupancy에서 shared
  LDS 명령어가 v3b 대비 2.5배로 늘어 issue efficiency가 나쁜 상태였다
  (scheduler 1.8사이클당 1회 issue, eligible warp 1.45).
- bottleneck after (NCU 실측, 4096³, `cp_async_pipe` 커널,
  `ncu_cp_async_pipe_details.csv`): register 92 reg/thread(v4a보다 10
  적음), Block Limit Registers=2, Theoretical Occupancy 33.33%(achieved
  32.38%) — v4a와 동일, register가 더 줄어도 occupancy 상한은 안 바뀐다.
  Compute(SM) 59.50%(v4a 54.32%), FMA 45.6%(v4a 42.6%)로 소폭 개선.
  결정적으로 STS(shared-store) 명령어가 **0건**(v4a는 8,388,608건) —
  cp.async가 global→shared 쓰기를 완전히 대체했기 때문이다. 그 결과
  MIO-throttle stall ratio가 2.29(v4a) → 1.64로 뚜렷이 줄었다.

  다만 shared LDS는 348,102,656건으로 v4a(335,544,320)보다 오히려 소폭
  많다(prologue의 별도 로드 때문으로 추정) — v4a에서 지적한 "regM
  스칼라-로드" 문제(`As`가 `[BM][BK]`라 float4 벡터화 불가)는 전혀
  해결되지 않고 그대로 남아 있다. shared-load bank conflict(3.2-way,
  49.86% wavefront), uncoalesced shared access(51%), uncoalesced global
  access(18%)도 v4a와 사실상 동일. Scheduler는 1.7사이클당 1회 issue,
  eligible warp 1.43/scheduler(v4a 1.45와 거의 동일) — issue 효율
  자체는 거의 개선되지 않았다.
- why it worked (or didn't): cp.async는 설계 의도대로 정확히 작동했다 —
  STS를 없애고 MIO throttle을 낮췄다(2.29→1.64). 하지만 이 latency-hiding
  이득은 v4a가 만든 진짜 병목(레이아웃 전환으로 인한 regM 벡터화 상실 →
  shared LDS 2.5배 증가)을 전혀 건드리지 못한다 — 그 병목은 compute
  phase의 읽기 패턴 문제이지 global→shared 전송 메커니즘과는 무관하기
  때문이다. 그래서 v4는 v4a보다 소폭 빠르지만(+6.6%), 벡터화된 read를
  유지한 v3b에는 여전히 크게 못 미친다.
- surprises: (1) misaligned address 에러의 진짜 원인은 "정렬"이 아니라
  인덱스 계산 실수(스케일 안 된 `innerColA`/`innerColB`를 그대로 씀)였다
  — cp.async의 16B 정렬 요구가 없었다면 조용히 틀린 값을 읽는
  correctness 버그로 남았을 것이다(디버깅 관점에서는 misaligned fault로
  바로 죽어준 게 오히려 다행이었다). (2) register가 92까지 더 줄어도
  occupancy 상한은 꼼짝 안 한다 — v3b부터 이어지는 "128 reg/thread
  문턱 안에서는 register 수가 occupancy에 영향 없다"는 관찰이 v4에서도
  유지된다. (3) 레이아웃 문제를 고치지 않는 한 cp.async 자체의 이득은
  상한이 뚜렷하다 — 다음 실험은 `As`를 v3b식 전치 레이아웃으로 되돌리며
  cp.async를 유지하는 조합(예: 전치된 목적지에 4번의 scalar 16B async
  copy로 흩뿌리기)이어야 한다.
- evidence: [NCU details](ncu_cp_async_pipe_details.csv)
