# RTX 3090 = Ampere GA102 = sm_86. Override on the CLI if you build elsewhere:
#   make ARCH=sm_75
ARCH    ?= sm_86
NVCC    ?= nvcc
NVFLAGS := -O3 -Xptxas -v -arch=$(ARCH) -std=c++17 -I../common -Iinclude -lineinfo
LDLIBS  := -lcublas

VERBOSE ?= 0
ifeq ($(VERBOSE),1)
  NVFLAGS += -Xptxas -v
endif


SRC := src/v0_cublas.cu src/v1_naive.cu src/v2_smem_tiling.cu \
       src/v3_register_tile.cu src/v3b_vectorize.cu src/v4a_doublebuffer.cu src/v4_cp_async_pipe.cu src/v4b_doublebuffer_tp.cu src/v4c_warptiling.cu src/v4d_wtdb.cu src/v4e_wtnt.cu src/v4f_db_tp_pad.cu src/v4g_db_tp_swizzle.cu src/v4g2.cu src/v4h_a_tp_b_async.cu src/v4g3.cu src/v4i.cu src/v4j.cu src/v4k.cu src/v4g2b.cu \
       driver/bench.cu

bench: $(SRC)
	$(NVCC) $(NVFLAGS) $(SRC) -o bench $(LDLIBS)

# v5 tensor-core ladder: FP32 cuBLAS baseline + the TF32/BF16 baselines and
# kernels, in a separate binary (bench.cu stays FP32-only). Add v5 kernel
# sources here as they land.
TC_SRC := src/v0_cublas.cu src/v0_cublas_tc.cu src/v5n1_wmma.cu \
          src/v5n2_smem_tiling.cu src/v5n3_warp_tile.cu src/v5n4_vec_pad.cu \
          src/v5n5_doublebuffer.cu src/v5n6_cp_async.cu src/v5s_wmma.cu src/v5m_mma.cu \
          driver/bench_tc.cu

# headers bench_tc depends on: rebuild when a launcher table or helper changes
TC_HDR := include/gemm.h include/gemm_tc.h ../common/common.cuh

bench_tc: $(TC_SRC) $(TC_HDR)
	$(NVCC) $(NVFLAGS) $(TC_SRC) -o bench_tc $(LDLIBS)

# default 2048^3 run
.PHONY: run bench-sizes profile profile-tc sanitize clean micro-tc profile-micro-tc
run: bench
	./bench

# a few shapes for the README table
bench-sizes: bench
	@for s in 1024 2048 4096; do ./bench $$s $$s $$s; echo; done

# profile one kernel with Nsight Compute, e.g.:
#   make profile KERNEL=naive_sgemm
#   make profile KERNEL=smem_sgemm
KERNEL ?= naive_sgemm
profile: bench
	ncu --set full -k $(KERNEL) -c 1 ./bench 1024 1024 1024

# v5-0: profile one kernel per precision at 4096^3 (skipping the 10 warmup
# launches) into docs/ncu/. Matches on the full demangled name because cuBLAS
# wraps CUTLASS kernels in a function whose base name is just "Kernel2".
TC_PROFILE := fp32:simt_sgemm tf32:s1688gemm bf16:s16816gemm_bf16
profile-tc: bench_tc
	@for p in $(TC_PROFILE); do \
	  ncu --set full --kernel-name-base demangled -k "regex:$${p#*:}" \
	      --launch-skip 10 -c 1 -f -o docs/ncu/v5-0_cublas_$${p%%:*}_4096 \
	      ./bench_tc 4096 4096 4096 1; \
	done

sanitize: bench
	compute-sanitizer --tool memcheck --error-exitcode 1 ./bench 1024 1024 1024 3

# tensor-pipe utilization ceiling microbenchmark: no memory traffic, just a
# closed loop of independent mma.sync ops per precision. See micro/tc_ceiling.cu.
micro-tc: micro/tc_ceiling

micro/tc_ceiling: micro/tc_ceiling.cu
	$(NVCC) $(NVFLAGS) $< -o $@

# one warmup + one timed launch per variant (micro/tc_ceiling's single-variant
# CLI mode), profiled with --launch-skip 1 -c 1 so ncu lands on the timed
# launch, not the warmup.
# The stall metric is the per-warp-active percentage: the
# smsp__average_warp_latency_issue_stalled_* counters are cumulative ratios
# whose .pct/.ratio rollups are not 0-100%.
TC_CEILING_VARIANTS := f16f16 f16f32 bf16f32 tf32f32
MICRO_METRICS := sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor_op_hmma.sum,gpu__time_duration.sum,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
profile-micro-tc: micro/tc_ceiling
	@for v in $(TC_CEILING_VARIANTS); do \
	  echo "=== $$v ==="; \
	  ncu --metrics $(MICRO_METRICS) --launch-skip 1 -c 1 \
	      ./micro/tc_ceiling $$v; \
	done

clean:
	rm -f bench bench_tc micro/tc_ceiling
