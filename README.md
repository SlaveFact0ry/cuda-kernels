# cuda-kernels

Optimizing sequential-inference kernels on an **RTX 3090 (Ampere, `sm_86`)**
against vendor speed-of-light, with the bottleneck at each step measured in
Nsight Compute. The goal across every chapter is not "a fast kernel" but a
reproducible account of *which hardware resource is the bottleneck and which
mechanism resolves it* — the portable judgment that transfers across kernels
(and, later, across architectures).

Part of a research line on **memory-hierarchy-aware acceleration on consumer
GPUs**: profiling where autoregressive/sequential inference actually stalls, and
closing the gap rung by rung.

## Chapters

| chapter | scope | status |
|---------|-------|--------|
| [gemm/](gemm/) | naive → smem → register tile → cp.async pipeline → WMMA | v1–v3b landed; v4–v5 stubs |
| [flash-attn/](flash-attn/) | FlashAttention-2 kernel + profile, sm_86 | planned (Aug 2026) |
| [kv-cache/](kv-cache/) | KV cache + speculative-decoding profiling | planned (Sep–Oct 2026) |

## Methodology

Every chapter follows the same loop: **measure → model (roofline) → hypothesize →
change → verify.** A vendor library (cuBLAS / FlashAttention-2) is the
speed-of-light baseline; the gap to it is the optimization target, and the
dominant limiter is identified from NCU counters *before* the next change. Each
optimization step is a separate, preserved commit so the history shows the
reasoning, not just the endpoint.

## Environment

CUDA toolkit (nvcc + cuBLAS), **Nsight Compute 2024.1.1**, **Nsight Systems
2023.4.4**, RTX 3090 (`sm_86`), driver 550.163.01. Architecture is a build
flag, not baked into any name: `make ARCH=sm_xx`. The analysis is written for
Ampere, where the full toolbox (async copy + tensor cores) is available.

The 3090 doubles as the display GPU, so boost clocks drift with thermal/desktop
load. `%SoL` figures are relatively stable (same-session ratio to the cuBLAS
run), but absolute GFLOP/s numbers are **not** clock-locked -- treat them as
indicative, not reproducible bit-for-bit. Lock clocks
(`sudo nvidia-smi -lgc <base_clock>`) before measurements that need to compare
precisely across sessions.

## Layout

```
common/     shared CUDA utilities (timing, verify, roofline helpers)
gemm/       GEMM optimization ladder  (see gemm/README.md)
flash-attn/ FlashAttention-2 chapter  (planned)
kv-cache/   KV cache + spec decoding  (planned)
docs/       per-chapter bottleneck logs — the running record that feeds writeups
```

## Build

Each chapter is self-contained; build from its own directory:

```bash
cd gemm && make && ./bench
```

## License

MIT — see [LICENSE](LICENSE).
