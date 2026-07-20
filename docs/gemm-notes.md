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
- evidence: [NCU summary](smem.csv) · [NCU screenshot](smem_sgemm.png)
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
