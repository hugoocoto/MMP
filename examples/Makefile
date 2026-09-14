CXX = mpicxx
MPIRUN = mpirun

NP = 8
BENCH_CSV ?= 0
TEST_RUNS ?= 10

CXXFLAGS = -O2 -march=native -Wall -std=c++20
CPPFLAGS = -I../malleable/include -DBENCH_CSV=$(BENCH_CSV) -MMD -MP
LDLIBS = -lpthread -ldl

BUILD_DIR = build
LIB = ../build/libmalleable.a

# Sources without main() are resize-policy plugins (.so); programs including
# malleable.hpp link against the library; the rest are plain MPI baselines.
SRCS := $(wildcard *.cpp)
PLUGIN_SRCS := $(shell grep -L '^int main' $(SRCS))
MAL_SRCS := $(shell grep -l '"malleable.hpp"' $(filter-out $(PLUGIN_SRCS),$(SRCS)))
PLAIN_SRCS := $(filter-out $(PLUGIN_SRCS) $(MAL_SRCS),$(SRCS))

PLUGINS = $(PLUGIN_SRCS:%.cpp=$(BUILD_DIR)/%.so)
MAL_BINS = $(MAL_SRCS:%.cpp=$(BUILD_DIR)/%)
PLAIN_BINS = $(PLAIN_SRCS:%.cpp=$(BUILD_DIR)/%)

export MAL_RESIZE_ENABLED ?= 1
export MAL_MALLEABILITY_ENABLED ?= 1
export MAL_LOAD_BALANCING_ENABLED ?= 1
export MAL_INITIAL_SIZE ?= 1
export MAL_AFFINITY ?= 0
export MAL_EPOCH_INTERVAL_MS ?= 100
export MAL_LOG_LEVEL ?= DEBUG
export MAL_EPOCH_CHANGE_MODE ?= 1
export MAL_TIMING ?= 0
export MAL_LOG_ALL_RANKS ?= 1

.PHONY: all test clean re FORCE

all: $(PLUGINS) $(MAL_BINS) $(PLAIN_BINS)

$(LIB): FORCE
	$(MAKE) -C .. all

$(MAL_BINS): $(BUILD_DIR)/%: $(BUILD_DIR)/%.o $(LIB)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -rdynamic -o $@

$(PLAIN_BINS): $(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(PLUGINS): $(BUILD_DIR)/%.so: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -shared $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# make run_<example> [NP=..] [ARGS=..]; per-example defaults go in ENV_<example> / ARGS_<example>.
TRANSPARENT = MAL_RESIZE_ENABLED=0 MAL_LOAD_BALANCING_ENABLED=0 MAL_INITIAL_SIZE=$(NP)
ENV_montecarlo = $(TRANSPARENT)
ENV_vector = $(TRANSPARENT)
ENV_matmat = $(TRANSPARENT)
ENV_matvec = $(TRANSPARENT)
ENV_sparse = $(TRANSPARENT)

NORM_N ?= 4000
ENV_normalize = MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=0 MAL_START_AT_UNIVERSE=1 MAL_INITIAL_SIZE=$(NP) MAL_AFFINITY=0
ARGS_normalize = --n=$(NORM_N)
ARGS_normalize_normal = --n=$(NORM_N)

LD_PERIOD ?= 6
LD_N ?= 8000
LD_GRID ?= 96
LD_FAST_STEPS ?= 1
LD_SLOW_STEPS ?= 12
LD_AMP ?= 100
LD_JITTER ?= 3
ARGS_life_dynamic = --period_sec=$(LD_PERIOD) --n=$(LD_N) --grid=$(LD_GRID) --fast_steps=$(LD_FAST_STEPS) --slow_steps=$(LD_SLOW_STEPS) --amp_pct=$(LD_AMP) --jitter_pct=$(LD_JITTER)

SPMV_FILE ?=
SPMV_SYNTH ?= 200000
SPMV_AVGDEG ?= 20
SPMV_PATTERN ?= clustered
SPMV_CONTRAST ?= 8
SPMV_CLUSTER_FRAC ?= 0.25
SPMV_REPS ?= 2000
ifeq ($(strip $(SPMV_FILE)),)
SPMV_ARGS = --synthetic=$(SPMV_SYNTH) --avg_deg=$(SPMV_AVGDEG) --pattern=$(SPMV_PATTERN) --contrast=$(SPMV_CONTRAST) --cluster_frac=$(SPMV_CLUSTER_FRAC) --reps=$(SPMV_REPS)
else
SPMV_ARGS = --file=$(SPMV_FILE) --reps=$(SPMV_REPS)
endif
ENV_spmv_imbal = MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=1 MAL_START_AT_UNIVERSE=1 MAL_INITIAL_SIZE=$(NP) MAL_AFFINITY=0 MAL_EPOCH_INTERVAL_MS=50 MAL_EPOCH_CHANGE_MODE=1
ENV_spmv_csr = MAL_MALLEABILITY_ENABLED=1 MAL_RESIZE_ENABLED=1 MAL_LOAD_BALANCING_ENABLED=1 MAL_START_AT_UNIVERSE=1 MAL_INITIAL_SIZE=$(NP) MAL_AFFINITY=0 MAL_EPOCH_INTERVAL_MS=60
ARGS_spmv_imbal = $(SPMV_ARGS)
ARGS_spmv_csr = $(SPMV_ARGS)
ARGS_spmv_imbal_normal = $(SPMV_ARGS)

run_montecarlo_plugin: $(PLUGINS)

run_%: $(BUILD_DIR)/%
	$(ENV_$*) $(MPIRUN) -n $(NP) --bind-to none ./$< $(ARGS_$*) $(ARGS)

VOTE_NP = 8
VOTE_ENV = MAL_LOAD_BALANCING_ENABLED=0 MAL_EPOCH_INTERVAL_MS=25

# check <label> <expected "[RESULT] ... OK" name> [VAR=value...] <command...>
test: all
	@total=$(TEST_RUNS); pass=0; fail=0; \
	check() { \
		label=$$1; expect=$$2; shift 2; \
		out=`env "$$@" 2>&1`; \
		if echo "$$out" | grep -q "\[RESULT\] $$expect OK"; then echo "  [OK] $$label"; else echo "  [FAIL] $$label"; iter_ok=0; fi; \
	}; \
	run="$(MPIRUN) -n $(NP) --bind-to none"; \
	vote="$(VOTE_ENV) $(MPIRUN) -n $(VOTE_NP) --bind-to none ./$(BUILD_DIR)/vote_consensus"; \
	for iter in $$(seq 1 $$total); do \
		echo ""; \
		echo "=== Iteration $$iter/$$total ==="; \
		iter_ok=1; \
		check "vector" "vector" $$run ./$(BUILD_DIR)/vector; \
		check "vector collapse" "vector" $$run ./$(BUILD_DIR)/vector collapse; \
		check "matvec" "mat-vec" $$run ./$(BUILD_DIR)/matvec; \
		check "matmat" "mat-mat" $$run ./$(BUILD_DIR)/matmat; \
		check "montecarlo" "montecarlo" $$run ./$(BUILD_DIR)/montecarlo; \
		check "vote unanimous" "vote unanimous" MAL_INITIAL_SIZE=4 $$vote unanimous --expect=6 --commits=1; \
		check "vote outlier" "vote outlier" MAL_INITIAL_SIZE=4 $$vote outlier --expect=6 --commits=1; \
		check "vote split" "vote split" MAL_INITIAL_SIZE=4 $$vote split --expect=4 --commits=0; \
		check "vote split quorum=0.75" "vote split" MAL_INITIAL_SIZE=4 MAL_RESIZE_QUORUM=0.75 $$vote split --expect=6 --commits=1; \
		check "vote turnout abstain=4" "vote turnout" MAL_INITIAL_SIZE=8 $$vote turnout --abstain=4 --expect=6 --commits=1; \
		check "vote turnout abstain=5" "vote turnout" MAL_INITIAL_SIZE=8 $$vote turnout --abstain=5 --expect=8 --commits=0; \
		check "vote rebalance" "vote rebalance" MAL_INITIAL_SIZE=4 $$vote rebalance --expect=4 --commits=1; \
		check "vote tie_grow" "vote tie_grow" MAL_INITIAL_SIZE=4 $$vote tie_grow --expect=5 --commits=1; \
		check "vote tie_shrink" "vote tie_shrink" MAL_INITIAL_SIZE=8 $$vote tie_shrink --expect=6 --commits=1; \
		if [ $$iter_ok -eq 1 ]; then pass=$$((pass+1)); echo "  => PASSED"; else fail=$$((fail+1)); echo "  => FAILED"; fi; \
	done; \
	echo ""; \
	echo "============================================"; \
	echo "  SUMMARY  ($$total iterations)"; \
	echo "--------------------------------------------"; \
	echo "  Passed : $$pass / $$total"; \
	echo "  Failed : $$fail / $$total"; \
	echo "============================================"; \
	if [ $$fail -eq 0 ]; then echo "  RESULT : ALL ITERATIONS OK"; else echo "  RESULT : $$fail ITERATION(S) FAILED"; fi; \
	echo "============================================"; \
	[ $$fail -eq 0 ]

clean:
	$(RM) -r $(BUILD_DIR)

re: clean
	$(MAKE) all

-include $(wildcard $(BUILD_DIR)/*.d)
