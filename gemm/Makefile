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
       src/v3_register_tile.cu src/v3b_vectorize.cu src/v4a_doublebuffer.cu src/v4_cp_async_pipe.cu src/v4b_doublebuffer_tp.cu src/v4c_warptiling.cu src/v4d_wtdb.cu src/v4e_wtnt.cu src/v4f_db_tp_pad.cu src/v4g_db_tp_swizzle.cu src/v4g2.cu src/v4h_a_tp_b_async.cu src/v4g3.cu src/v4i.cu src/v4j.cu src/v5_wmma_tc.cu \
       driver/bench.cu

bench: $(SRC)
	$(NVCC) $(NVFLAGS) $(SRC) -o bench $(LDLIBS)

# default 2048^3 run
.PHONY: run bench-sizes profile sanitize clean
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

sanitize: bench
	compute-sanitizer --tool memcheck --error-exitcode 1 ./bench 1024 1024 1024 3

clean:
	rm -f bench
