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
TC_SRC := src/v0_cublas.cu src/v0_cublas_tc.cu \
          driver/bench_tc.cu

bench_tc: $(TC_SRC)
	$(NVCC) $(NVFLAGS) $(TC_SRC) -o bench_tc $(LDLIBS)

# default 2048^3 run
.PHONY: run bench-sizes profile profile-tc sanitize clean
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

clean:
	rm -f bench bench_tc
