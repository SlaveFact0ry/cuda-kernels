#!/usr/bin/env python3
"""Roofline for the RTX 3090 (Ampere GA102, sm_86) with measured points overlaid.

Peak numbers are DEFAULTS — verify yours (deviceQuery / measured STREAM-style
bandwidth) and update. Arithmetic intensity for a square GEMM (NxNxN, fp32) is
roughly  AI = 2*N^3 / (3*N^2*4 bytes) = N/6  FLOP/byte, i.e. it climbs with N,
which is exactly why GEMM moves from memory-bound (small N) toward compute-bound
(large N). Note the ridge sits HIGHER on the 3090 (~38) than on Turing (~22),
because Ampere's fp32 throughput grew more than its bandwidth did — so you need
a larger N before fp32 GEMM becomes compute-bound. Add your (AI, GFLOP/s) points
from the bench output to `points`.

Usage:  python scripts/roofline.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PEAK_FP32_GFLOPS = 35600.0   # ~35.6 TFLOPS fp32 (3090 boost, 2x FP32 datapath) — verify
PEAK_BW_GBs      = 936.0     # ~936 GB/s (GDDR6X, 384-bit) — verify

# (label, arithmetic_intensity_flop_per_byte, achieved_GFLOPs)
points = [
    ("v0_cublas 2048", 2048/6, 22247.4),
    ("v0_cublas 4096", 4096/6, 22873.3),
    ("v1_naive 2048",  2048/6,  2213.1),
    ("v1_naive 4096",  4096/6,  2151.2),
    ("v2_smem 2048",   2048/6,  2985.1),
    ("v2_smem 4096",   4096/6,  2910.8),
    ("v3_regtile 2048", 2048/6, 8791.7),
    ("v3_regtile 4096", 4096/6, 9431.4),
]

ai = np.logspace(-1, 4, 400)
roof = np.minimum(PEAK_FP32_GFLOPS, PEAK_BW_GBs * ai)
ridge = PEAK_FP32_GFLOPS / PEAK_BW_GBs

plt.figure(figsize=(7, 5))
plt.loglog(ai, roof, "k-", lw=2, label="roofline")
plt.axhline(PEAK_FP32_GFLOPS, ls="--", c="grey", lw=1)
plt.axvline(ridge, ls=":", c="grey", lw=1, label=f"ridge AI={ridge:.1f}")
for label, x, y in points:
    plt.plot(x, y, "o")
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4))

plt.xlabel("arithmetic intensity (FLOP/byte)")
plt.ylabel("performance (GFLOP/s)")
plt.title("RTX 3090 (Ampere, sm_86) roofline")
plt.legend(); plt.grid(True, which="both", ls=":", alpha=.4)
plt.tight_layout(); plt.savefig("results/roofline.png", dpi=140)
print("wrote results/roofline.png")
