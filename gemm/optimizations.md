# GEMM Kernel Optimizations

Terse reference of which techniques each kernel version applies. See README.md / docs/gemm-notes.md for perf numbers and NCU evidence.

| version | optimizations |
|---|---|
| v0_cublas | cuBLAS 레퍼런스 |
| v1_naive | naive |
| v2_smem_tiling | shared mem tiling |
| v3_register_tile | + register tiling |
| v3b_vectorize | + float4 vectorize + A transpose |
| v4a_doublebuffer | reg tiling + vectorize + double buffer (A no transpose) |
| v4b_doublebuffer_tp | v4a + A transpose |
| v4c_warptiling | v3b + warptiling |
| v4d_wtdb | v4c + double buffer |
| v4e_wtnt | v4c, A no transpose |
| v4f_db_tp_pad | double buffer + transpose + B padding |
| v4g_db_tp_swizzle | double buffer + transpose + B swizzle |
| v4g2 | double buffer + transpose + B swizzle v2 |
| v4g3 | v4g2 + warptiling |
| v4h_a_tp_b_async | double buffer + transpose + cp.async(B) |
| v4i | v4g2 + cp.async(B) |
| v4j | v4g3 + cp.async(B) |
| v4_cp_async_pipe | double buffer + cp.async(A&B) |
| v5_wmma_tc | WMMA — 미구현 스텁 |
