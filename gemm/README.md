# gemm — single-precision GEMM optimization ladder

Chapter of [`cuda-kernels`](../README.md). Step-by-step optimization of fp32 GEMM
on an **RTX 3090 (Ampere GA102, `sm_86`)**, profiled at every stage against the
cuBLAS speed-of-light baseline. The goal is not "a fast matmul" — it is a clean,
measured account of *which* bottleneck dominates at each rung and *why*, up to a
cp.async-fed, tensor-core kernel that mirrors what cuBLAS/CUTLASS do internally.

> Methodology: **profile before optimizing.** Each version is verified against
> cuBLAS, timed, and analyzed with Nsight Compute. The gap to the SoL roofline
> is the optimization target; the limiter is identified before the next change.

## Results (4096³, fp32)

Filled in as each rung lands. `%SoL` = achieved GFLOP/s ÷ cuBLAS GFLOP/s.

| version             | GFLOP/s | %SoL | dominant limiter (from NCU) |
|---------------------|--------:|-----:|-----------------------------|
| v0 cuBLAS (SoL)     | 20543.9 | 100  | reference                   |
| v1 naive            |  1802.9 |  8.8 | L1TEX/LSU pipe saturation (NOT DRAM-bound — DRAM throughput only 1.6%; one uncoalesced global load per thread per iter) |
| v2 smem tiling      |  2237.8 | 10.9 | MIO pipe saturation (shared-mem load per iter) + occupancy regression (66.7%, 1024-thread block) |
| v3 register tiling  |  7490.5 | 36.5 | Register-limited occupancy (96 reg/thread → 33.3% theoretical, 2 blocks/SM) starves latency-hiding — short+long-scoreboard stalls dominate; NOT compute-bound (FMA pipe ~10%), + 50% shared-load bank conflicts (3.2-way) |
| v3b vectorized loads| 15795.3 | 76.9 | Still register-limited occupancy (113 reg/thread, 33.3%, unchanged from v3), compute-bound-adjacent (Compute(SM) 60.0%, FMA pipe 56.5%); global→shared vectorization halved shared-store conflicts (4.5-way → 2.4-way) while shared-load conflicts stayed at 5.0-way |
| v4 cp.async pipeline|    —    |  —   | _TODO: stalls hidden?_      |
| v5 WMMA tensor core |    —    |  —   | _TODO (fp16/tf32)_          |

Measured with GPU clocks locked (`nvidia-smi -lgc`) for cross-session
reproducibility; see the root README's clock-drift caveat for why that
matters and why earlier indicative numbers aren't directly comparable.

Roofline plot: [`../docs/roofline.png`](../docs/roofline.png) (regenerate with
`scripts/roofline.py`; x-axis is algorithmic AI = N/6, assuming full reuse —
not NCU-measured DRAM traffic).
Per-stage analysis: [`../docs/gemm-notes.md`](../docs/gemm-notes.md).

## Build & run

```bash
# from this directory (gemm/)
make                 # builds ./bench for sm_86 (override: make ARCH=sm_75)
./bench              # default 2048^3
./bench 4096 4096 4096 100
make bench-sizes     # 1024 / 2048 / 4096
make profile KERNEL=naive_sgemm   # NCU on one kernel by name (default: naive_sgemm)
make profile KERNEL=smem_sgemm
make sanitize        # compute-sanitizer memcheck on v1-v3b at 1024^3
```

v1 handles arbitrary shapes. v2+ kernels require tile-multiple shapes; the
launcher rejects other M/N/K explicitly (prints to stderr, row shows `skip`).

Shared timing/verify utilities live in [`../common/common.cuh`](../common/common.cuh);
the Makefile adds `-I../common`. Requires CUDA toolkit (nvcc + cuBLAS) and an
Ampere-or-newer GPU for the v4/v5 features (v1–v3 build anywhere).

## The optimization ladder

1. **v1 naive** — one thread per C element, no reuse. Memory-bound baseline.
2. **v2 shared-memory tiling** — stage A/B tiles in smem so each load is reused
   block-wide; cut global traffic, watch bank conflicts and coalescing. Ampere
   lets you opt into ~100 KB smem/block for bigger tiles.
3. **v3 register/thread tiling** — each thread computes an 8×8 micro-tile in
   registers; raises arithmetic intensity. Occupancy-vs-ILP trade-off lives here.
   - **v3b vectorized loads** — float4 loads for the shared-memory reads feeding
     regM/regN (2 vectorized loads/operand instead of 8 scalar loads), and
     float4 loads for the global→shared A/B staging (1 vectorized load/thread
     per tile instead of a strided scalar loop); same TM=TN=8 register
     footprint family as v3 (96 → 113 reg/thread, occupancy unchanged).
4. **v4 cp.async software pipeline** — float4 async global→shared copies with a
   multi-stage (2–4) buffer, hiding global latency behind compute. *This is the
   rung Ampere unlocks — impossible on the previous Turing card.* Same structure
   CUTLASS uses.
5. **v5 WMMA tensor cores** — fp16 (16×16×16) / tf32 (16×16×8), fp32 accumulate;
   lifts the compute roof to ~142 TFLOP/s, flipping the limiter back to feeding
   the cores. Strongest variant = cp.async-fed WMMA.

v1–v3b are implemented; **v4–v5 are stubs with specs in the source
comments** — implement one commit at a time so the history shows the progression.

## Layout

```
include/   gemm.h (version registry)   [common.cuh is shared at ../common]
src/       v0_cublas + v1..v5 kernels
driver/    bench.cu — verify + time each version vs cuBLAS
scripts/   roofline.py
results/   generated plots (gitignored)
```

## Hardware / peak figures

RTX 3090, `sm_86`: ~35.6 TFLOP/s fp32, ~142 TFLOP/s fp16 (tensor core), ~936
GB/s. **Verify these for your card** and update `scripts/roofline.py`. The fp32
ridge point (AI ≈ peak/BW ≈ 38 FLOP/byte) sits higher than on Turing, because
Ampere's fp32 throughput grew more than its bandwidth — so square fp32 GEMM
needs a larger N before it becomes compute-bound. That shift is itself part of
the story.
