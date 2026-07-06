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

## v1_naive (YYYY-MM-DD)
- change: baseline, one thread per C element, no reuse.
- result: TODO
- bottleneck: TODO (predict memory-bound; CONFIRM with NCU before believing it)
