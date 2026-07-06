# kv-cache — KV cache + speculative decoding profiling (planned)

Chapter of [`cuda-kernels`](../README.md). **Planned: September–October 2026.**

Profiling the memory-bound decode phase of autoregressive inference on `sm_86`:
KV cache access patterns and the kernel-level cost of speculative decoding. This
is the empirical core of the Track A study ("When Draft Meets Diffusion") — the
claimed gap is that prior speculative-decoding work is A100/H100 + algorithm-level
with no measured kernel profiling on consumer GPUs.

Planned measurements (subject to design):
- decode-step kernels: measured arithmetic intensity, HBM utilization, occupancy,
  L1/L2 hit rate.
- speculative-decoding cases: C1 (AR draft + AR target), C2 (DLM draft + AR
  target), C3 (DLM step sweep 1/2/4/8).
- draft model: TBD (see research plan; SEDD-Absorbing Small + GPT-2 XL under
  consideration).

Reference reading (this chapter): llama.cpp `ggml-cuda` KV-cache + quantized
matmul paths, deferred here from the GEMM chapter. Deliverable: bottleneck log in
[`../docs`](../docs/) feeding the arXiv draft.
