CXX = mpicxx
AR = ar
MPIRUN = mpirun

NP = 8
BENCH_CSV ?= 0
TEST_RUNS ?= 10
NORM_N ?= 4000

USE_PAPI ?= 0
PAPI_PREFIX ?=

CXXFLAGS = -O2 -march=native -Wall -std=c++20 -Imalleable/include -DBENCH_CSV=$(BENCH_CSV)
LIBS = -lpthread -ldl

RD_DYN_PERIOD ?= 6
RD_DYN_N ?= 50000
RD_DYN_FAST_US ?= 2000
RD_DYN_SLOW_US ?= 12000
RD_DYN_AMP ?= 60
RD_DYN_JITTER ?= 3

export MAL_RESIZE_ENABLED=1
export MAL_MALLEABILITY_ENABLED=1
export MAL_LOAD_BALANCING_ENABLED=1
export MAL_INITIAL_SIZE=1
export MAL_AFFINITY=0
export MAL_EPOCH_INTERVAL_MS=100
export MAL_LOG_LEVEL=DEBUG
export MAL_EPOCH_CHANGE_MODE=1
export MAL_TIMING=0
export MAL_LOG_ALL_RANKS=1

ifeq ($(USE_PAPI), 1)

	CXXFLAGS += -DMAL_USE_PAPI

	ifneq ($(strip $(PAPI_PREFIX)),)

		CXXFLAGS += -I$(PAPI_PREFIX)/include
		LIBS += -L$(PAPI_PREFIX)/lib -Wl,-rpath,$(PAPI_PREFIX)/lib

	endif

	ifneq ($(strip $(PFM_PREFIX)),)

		LIBS += -L$(PFM_PREFIX)/lib -Wl,-rpath,$(PFM_PREFIX)/lib

	endif

	LIBS += -lpapi

	ifneq ($(strip $(PFM_PREFIX)),)

		LIBS += -lpfm

	endif

endif

BUILD_DIR = build

LIB_MAL = $(BUILD_DIR)/libmalleable.a
LIB_SRCS = malleable/src/malleable.cpp
LIB_OBJS = $(LIB_SRCS:%.cpp=$(BUILD_DIR)/%.o)
LIB_DEPS = $(wildcard malleable/src/malleable_*.cpp malleable/include/*.hpp malleable/builtin_policies/*.cpp)

TARGET_MC = $(BUILD_DIR)/montecarlo
SRCS_MC = examples/montecarlo.cpp
OBJS_MC = $(SRCS_MC:%.cpp=$(BUILD_DIR)/%.o)

TARGET_VEC = $(BUILD_DIR)/vector
SRCS_VEC = examples/vector.cpp
OBJS_VEC = $(SRCS_VEC:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MAT = $(BUILD_DIR)/matmat
SRCS_MAT = examples/matmat.cpp
OBJS_MAT = $(SRCS_MAT:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MV = $(BUILD_DIR)/matvec
SRCS_MV = examples/matvec.cpp
OBJS_MV = $(SRCS_MV:%.cpp=$(BUILD_DIR)/%.o)

TARGET_SP = $(BUILD_DIR)/sparse
SRCS_SP = examples/sparse.cpp
OBJS_SP = $(SRCS_SP:%.cpp=$(BUILD_DIR)/%.o)

TARGET_LD = $(BUILD_DIR)/life_dynamic
SRCS_LD = examples/life_dynamic.cpp
OBJS_LD = $(SRCS_LD:%.cpp=$(BUILD_DIR)/%.o)

TARGET_RD = $(BUILD_DIR)/resize_demo
SRCS_RD = examples/resize_demo.cpp
OBJS_RD = $(SRCS_RD:%.cpp=$(BUILD_DIR)/%.o)

TARGET_CP = $(BUILD_DIR)/custom_policy
SRCS_CP = examples/custom_policy.cpp
OBJS_CP = $(SRCS_CP:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MCP  = $(BUILD_DIR)/montecarlo_plugin_policy.so
SRCS_MCP    = examples/montecarlo_plugin_policy.cpp
OBJS_MCP    = $(SRCS_MCP:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MCPL = $(BUILD_DIR)/montecarlo_plugin
SRCS_MCPL   = examples/montecarlo_plugin.cpp
OBJS_MCPL   = $(SRCS_MCPL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_SPMV = $(BUILD_DIR)/spmv_imbal
SRCS_SPMV = examples/spmv_imbal.cpp
OBJS_SPMV = $(SRCS_SPMV:%.cpp=$(BUILD_DIR)/%.o)

TARGET_SPMV_CSR = $(BUILD_DIR)/spmv_csr
SRCS_SPMV_CSR = examples/spmv_csr.cpp
OBJS_SPMV_CSR = $(SRCS_SPMV_CSR:%.cpp=$(BUILD_DIR)/%.o)

TARGET_NOISE = $(BUILD_DIR)/noise_bench
SRCS_NOISE = examples/noise_bench.cpp
OBJS_NOISE = $(SRCS_NOISE:%.cpp=$(BUILD_DIR)/%.o)

TARGET_NOISE_NORMAL = $(BUILD_DIR)/noise_bench_normal
SRCS_NOISE_NORMAL = examples/noise_bench_normal.cpp
OBJS_NOISE_NORMAL = $(SRCS_NOISE_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_NOISE_DAEMON = $(BUILD_DIR)/noise_daemon
SRCS_NOISE_DAEMON = examples/noise_daemon.cpp
OBJS_NOISE_DAEMON = $(SRCS_NOISE_DAEMON:%.cpp=$(BUILD_DIR)/%.o)

TARGET_SPMV_NORMAL = $(BUILD_DIR)/spmv_imbal_normal
SRCS_SPMV_NORMAL = examples/spmv_imbal_normal.cpp
OBJS_SPMV_NORMAL = $(SRCS_SPMV_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MC_NORMAL = $(BUILD_DIR)/montecarlo_normal
SRCS_MC_NORMAL = examples/montecarlo_normal.cpp
OBJS_MC_NORMAL = $(SRCS_MC_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_VEC_NORMAL = $(BUILD_DIR)/vector_normal
SRCS_VEC_NORMAL = examples/vector_normal.cpp
OBJS_VEC_NORMAL = $(SRCS_VEC_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MM_NORMAL = $(BUILD_DIR)/matmat_normal
SRCS_MM_NORMAL = examples/matmat_normal.cpp
OBJS_MM_NORMAL = $(SRCS_MM_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_MV_NORMAL = $(BUILD_DIR)/matvec_normal
SRCS_MV_NORMAL = examples/matvec_normal.cpp
OBJS_MV_NORMAL = $(SRCS_MV_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_SP_NORMAL = $(BUILD_DIR)/sparse_normal
SRCS_SP_NORMAL = examples/sparse_normal.cpp
OBJS_SP_NORMAL = $(SRCS_SP_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_HEAT_NORMAL = $(BUILD_DIR)/heat_normal
SRCS_HEAT_NORMAL = examples/heat_normal.cpp
OBJS_HEAT_NORMAL = $(SRCS_HEAT_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

TARGET_HEAT = $(BUILD_DIR)/heat
SRCS_HEAT = examples/heat.cpp
OBJS_HEAT = $(SRCS_HEAT:%.cpp=$(BUILD_DIR)/%.o)

TARGET_NORM = $(BUILD_DIR)/normalize
SRCS_NORM = examples/normalize.cpp
OBJS_NORM = $(SRCS_NORM:%.cpp=$(BUILD_DIR)/%.o)

TARGET_NORM_NORMAL = $(BUILD_DIR)/normalize_normal
SRCS_NORM_NORMAL = examples/normalize_normal.cpp
OBJS_NORM_NORMAL = $(SRCS_NORM_NORMAL:%.cpp=$(BUILD_DIR)/%.o)

DEPS =$(LIB_OBJS:.o=.d) $(OBJS_MC:.o=.d) $(OBJS_VEC:.o=.d) $(OBJS_MAT:.o=.d) $(OBJS_MV:.o=.d) $(OBJS_SP:.o=.d) $(OBJS_RD:.o=.d) $(OBJS_LD:.o=.d) $(OBJS_SPMV:.o=.d) $(OBJS_SPMV_NORMAL:.o=.d) $(OBJS_MC_NORMAL:.o=.d) $(OBJS_VEC_NORMAL:.o=.d) $(OBJS_MM_NORMAL:.o=.d) $(OBJS_MV_NORMAL:.o=.d) $(OBJS_SP_NORMAL:.o=.d)

.PHONY: all run_montecarlo run_vector run_vector_collapse run_normalize run_normalize_normal run_matmat run_matvec run_sparse run_resize_demo run_resize_demo_dynamic run_life_dynamic run_spmv_imbal run_spmv_csr run_spmv_imbal_normal run_montecarlo_normal run_vector_normal run_vector_normal_collapse run_matmat_normal run_matvec_normal run_sparse_normal run_custom_policy run_montecarlo_plugin test clean re

all: $(LIB_MAL) $(TARGET_MC) $(TARGET_VEC) $(TARGET_MAT) $(TARGET_MV) $(TARGET_SP) $(TARGET_RD) $(TARGET_LD) $(TARGET_SPMV) $(TARGET_SPMV_NORMAL) $(TARGET_MC_NORMAL) $(TARGET_VEC_NORMAL) $(TARGET_MM_NORMAL) $(TARGET_MV_NORMAL) $(TARGET_SP_NORMAL) $(TARGET_HEAT_NORMAL) $(TARGET_HEAT) $(TARGET_NORM) $(TARGET_NORM_NORMAL) $(TARGET_SPMV_CSR) $(TARGET_NOISE) $(TARGET_NOISE_NORMAL) $(TARGET_NOISE_DAEMON) $(TARGET_CP) $(TARGET_MCP) $(TARGET_MCPL)

$(LIB_MAL): Makefile $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(RM) $@
	$(AR) rcs $@ $(LIB_OBJS)

$(LIB_OBJS): $(LIB_DEPS)

$(TARGET_MC): $(LIB_MAL) $(OBJS_MC)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MC) $(LIB_MAL) $(LIBS)

$(TARGET_VEC): $(LIB_MAL) $(OBJS_VEC)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_VEC) $(LIB_MAL) $(LIBS)

$(TARGET_MAT): $(LIB_MAL) $(OBJS_MAT)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MAT) $(LIB_MAL) $(LIBS)

$(TARGET_MV): $(LIB_MAL) $(OBJS_MV)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MV) $(LIB_MAL) $(LIBS)

$(TARGET_SP): $(LIB_MAL) $(OBJS_SP)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SP) $(LIB_MAL) $(LIBS)

$(TARGET_RD): $(LIB_MAL) $(OBJS_RD)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_RD) $(LIB_MAL) $(LIBS)

$(TARGET_CP): $(LIB_MAL) $(OBJS_CP)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_CP) $(LIB_MAL) $(LIBS)

$(TARGET_MCP): $(SRCS_MCP) malleable/include/malleable.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -fPIC -shared -o $@ $(SRCS_MCP)

$(TARGET_MCPL): $(LIB_MAL) $(OBJS_MCPL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MCPL) $(LIB_MAL) $(LIBS) -rdynamic

$(TARGET_LD): $(LIB_MAL) $(OBJS_LD)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_LD) $(LIB_MAL) $(LIBS)

$(TARGET_SPMV): $(LIB_MAL) $(OBJS_SPMV)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SPMV) $(LIB_MAL) $(LIBS)

$(TARGET_SPMV_CSR): $(LIB_MAL) $(OBJS_SPMV_CSR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SPMV_CSR) $(LIB_MAL) $(LIBS)

$(TARGET_NOISE): $(LIB_MAL) $(OBJS_NOISE)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NOISE) $(LIB_MAL) $(LIBS)

$(TARGET_NOISE_NORMAL): $(OBJS_NOISE_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NOISE_NORMAL) $(LIBS)

$(TARGET_NOISE_DAEMON): $(OBJS_NOISE_DAEMON)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NOISE_DAEMON) $(LIBS)

$(TARGET_SPMV_NORMAL): $(OBJS_SPMV_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SPMV_NORMAL) $(LIBS)

$(TARGET_MC_NORMAL): $(OBJS_MC_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MC_NORMAL) $(LIBS)

$(TARGET_VEC_NORMAL): $(OBJS_VEC_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_VEC_NORMAL) $(LIBS)

$(TARGET_MM_NORMAL): $(OBJS_MM_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MM_NORMAL) $(LIBS)

$(TARGET_MV_NORMAL): $(OBJS_MV_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MV_NORMAL) $(LIBS)

$(TARGET_HEAT_NORMAL): $(OBJS_HEAT_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_HEAT_NORMAL) $(LIBS)

$(TARGET_HEAT): $(LIB_MAL) $(OBJS_HEAT)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_HEAT) $(LIB_MAL) $(LIBS)

$(TARGET_NORM): $(LIB_MAL) $(OBJS_NORM)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NORM) $(LIB_MAL) $(LIBS)

$(TARGET_NORM_NORMAL): $(OBJS_NORM_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_NORM_NORMAL) $(LIBS)

$(TARGET_SP_NORMAL): $(OBJS_SP_NORMAL)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_SP_NORMAL) $(LIBS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

TRANSPARENT = MAL_RESIZE_ENABLED=0 MAL_LOAD_BALANCING_ENABLED=0 MAL_INITIAL_SIZE=$(NP)

run_montecarlo: $(TARGET_MC)
	$(TRANSPARENT) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MC)

run_vector: $(TARGET_VEC)
	$(TRANSPARENT) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC)

run_vector_collapse: $(TARGET_VEC)
	$(TRANSPARENT) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC) collapse

run_normalize: $(TARGET_NORM)
	MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=0 MAL_START_AT_UNIVERSE=1 MAL_INITIAL_SIZE=$(NP) MAL_AFFINITY=0 $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_NORM) --n=$(NORM_N)

run_normalize_normal: $(TARGET_NORM_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_NORM_NORMAL) --n=$(NORM_N)

run_matmat: $(TARGET_MAT)
	$(TRANSPARENT) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MAT)

run_matvec: $(TARGET_MV)
	$(TRANSPARENT) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MV)

run_sparse: $(TARGET_SP)
	$(TRANSPARENT) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_SP)

run_resize_demo: $(TARGET_RD)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_RD)

run_custom_policy: $(TARGET_CP)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_CP)

run_montecarlo_plugin: $(TARGET_MCP) $(TARGET_MCPL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MCPL)

run_resize_demo_dynamic: $(TARGET_RD)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_RD) --dynamic=1 --period_sec=$(RD_DYN_PERIOD) --n=$(RD_DYN_N) --fast_us=$(RD_DYN_FAST_US) --slow_us=$(RD_DYN_SLOW_US) --amp_pct=$(RD_DYN_AMP) --jitter_pct=$(RD_DYN_JITTER)

LD_PERIOD ?= 6
LD_N ?= 8000
LD_GRID ?= 96
LD_FAST_STEPS ?= 1
LD_SLOW_STEPS ?= 12
LD_AMP ?= 100
LD_JITTER ?= 3

run_life_dynamic: $(TARGET_LD)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_LD) --period_sec=$(LD_PERIOD) --n=$(LD_N) --grid=$(LD_GRID) --fast_steps=$(LD_FAST_STEPS) --slow_steps=$(LD_SLOW_STEPS) --amp_pct=$(LD_AMP) --jitter_pct=$(LD_JITTER)

SPMV_FILE ?=
SPMV_SYNTH ?= 200000
SPMV_AVGDEG ?= 20
SPMV_PATTERN ?= clustered
SPMV_CONTRAST ?= 8
SPMV_CLUSTER_FRAC ?= 0.25
SPMV_REPS ?= 2000

ifeq ($(strip $(SPMV_FILE)),)
	SPMV_SRC_ARGS = --synthetic=$(SPMV_SYNTH) --avg_deg=$(SPMV_AVGDEG) --pattern=$(SPMV_PATTERN) --contrast=$(SPMV_CONTRAST) --cluster_frac=$(SPMV_CLUSTER_FRAC)
else
	SPMV_SRC_ARGS = --file=$(SPMV_FILE)
endif
	SPMV_ARGS = $(SPMV_SRC_ARGS) --reps=$(SPMV_REPS)

SPMV_MAL_ENV = MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=1 MAL_START_AT_UNIVERSE=1 MAL_INITIAL_SIZE=$(NP) MAL_AFFINITY=0 MAL_EPOCH_INTERVAL_MS=50 MAL_EPOCH_CHANGE_MODE=1

run_spmv_imbal: $(TARGET_SPMV)
	$(SPMV_MAL_ENV) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_SPMV) $(SPMV_ARGS)

SPMV_CSR_ENV = MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=1 MAL_START_AT_UNIVERSE=1 MAL_INITIAL_SIZE=$(NP) MAL_AFFINITY=0 MAL_EPOCH_INTERVAL_MS=60

run_spmv_csr: $(TARGET_SPMV_CSR)
	$(SPMV_CSR_ENV) $(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_SPMV_CSR) $(SPMV_ARGS)

run_spmv_imbal_normal: $(TARGET_SPMV_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_SPMV_NORMAL) $(SPMV_ARGS)

run_montecarlo_normal: $(TARGET_MC_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MC_NORMAL)

run_vector_normal: $(TARGET_VEC_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC_NORMAL)

run_vector_normal_collapse: $(TARGET_VEC_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC_NORMAL) collapse

run_matmat_normal: $(TARGET_MM_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MM_NORMAL)

run_matvec_normal: $(TARGET_MV_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MV_NORMAL)

run_sparse_normal: $(TARGET_SP_NORMAL)
	$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_SP_NORMAL)

test: all
	@total=$(TEST_RUNS); pass=0; fail=0; \
	for iter in $$(seq 1 $$total); do \
		echo ""; \
		echo "=== Iteration $$iter/$$total ==="; \
		iter_ok=1; \
		out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC) 2>&1`; \
		echo "$$out" | grep -q "\[RESULT\] vector OK" && echo "  [OK] vector" || { echo "  [FAIL] vector"; iter_ok=0; }; \
		out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_VEC) collapse 2>&1`; \
		echo "$$out" | grep -q "\[RESULT\] vector OK" && echo "  [OK] vector collapse" || { echo "  [FAIL] vector collapse"; iter_ok=0; }; \
		out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MV) 2>&1`; \
		echo "$$out" | grep -q "\[RESULT\] mat-vec OK" && echo "  [OK] matvec" || { echo "  [FAIL] matvec"; iter_ok=0; }; \
		out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MAT) 2>&1`; \
		echo "$$out" | grep -q "\[RESULT\] mat-mat OK" && echo "  [OK] matmat" || { echo "  [FAIL] matmat"; iter_ok=0; }; \
		out=`$(MPIRUN) -n $(NP) --bind-to none ./$(TARGET_MC) 2>&1`; \
		echo "$$out" | grep -q "\[RESULT\] montecarlo OK" && echo "  [OK] montecarlo" || { echo "  [FAIL] montecarlo"; iter_ok=0; }; \
		if [ $$iter_ok -eq 1 ]; then \
			pass=$$((pass+1)); echo "  => PASSED"; \
		else \
			fail=$$((fail+1)); echo "  => FAILED"; \
		fi; \
	done; \
	echo ""; \
	echo "============================================"; \
	echo "  SUMMARY  ($$total iterations)"; \
	echo "--------------------------------------------"; \
	echo "  Passed : $$pass / $$total"; \
	echo "  Failed : $$fail / $$total"; \
	echo "============================================"; \
	if [ $$fail -eq 0 ]; then \
		echo "  RESULT : ALL ITERATIONS OK"; \
	else \
		echo "  RESULT : $$fail ITERATION(S) FAILED"; \
	fi; \
	echo "============================================"; \
	[ $$fail -eq 0 ]

re: clean all

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)
