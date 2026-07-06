# flash-attn — FlashAttention-2 kernel + profile (planned)

Chapter of [`cuda-kernels`](../README.md). **Planned: August 2026.**

Hand-implemented FlashAttention-2 style attention on `sm_86`, built on the
shared-memory tiling and cp.async pipelining developed in [`../gemm`](../gemm/),
then profiled against a vendor FA2 baseline with the same measure→model→verify
loop.

Why this chapter exists: attention is the sequential-inference bottleneck this
research line targets. On the 3090 (unlike the earlier Turing card) FlashAttention
2 is available, so the vendor path is a fair speed-of-light reference.

Planned rungs (subject to profiling):
- naive attention (materialized scores) — memory baseline.
- online-softmax tiling (FA1-style) — remove the O(N²) materialization.
- cp.async-pipelined tiles (FA2-style) — hide global latency.
- tensor-core QKᵀ / PV — compute-bound target.

Deliverable: results table (% of vendor FA2) + roofline + bottleneck log in
[`../docs`](../docs/), same format as the GEMM chapter.
