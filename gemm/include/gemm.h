#pragma once

// Each kernel version is a launcher with the SAME signature.
// Row-major, single precision: C(MxN) = alpha * A(MxK) * B(KxN) + beta * C(MxN).
//
// Return true if the version is implemented and was launched, false to skip
// the row (still a stub, or the launcher rejected this M/N/K — e.g. v2+
// require tile-multiple shapes). The benchmark harness skips gracefully
// either way. This lets the repo always compile while you climb the ladder
// one commit at a time.
using GemmLauncher = bool (*)(int M, int N, int K, float alpha, float beta,
                              const float* dA, const float* dB, float* dC);

bool launch_v1_naive   (int, int, int, float, float, const float*, const float*, float*);
bool launch_v2_smem    (int, int, int, float, float, const float*, const float*, float*);
bool launch_v3_regtile (int, int, int, float, float, const float*, const float*, float*);
bool launch_v3b_vectorize(int, int, int, float, float, const float*, const float*, float*);
bool launch_v4_cp_async(int, int, int, float, float, const float*, const float*, float*);
bool launch_v5_wmma    (int, int, int, float, float, const float*, const float*, float*);

struct Version { const char* name; GemmLauncher fn; };

// The harness iterates this table. Add rows here as you implement versions.
inline const Version* version_table(int& count) {
  static const Version table[] = {
    {"v1_naive",        launch_v1_naive},
    {"v2_smem_tiling",  launch_v2_smem},
    {"v3_register_tile",launch_v3_regtile},
    {"v3b_vectorize",   launch_v3b_vectorize},
    {"v4_cp_async_pipe",launch_v4_cp_async},
    {"v5_wmma_tc",      launch_v5_wmma},
  };
  count = sizeof(table) / sizeof(table[0]);
  return table;
}
