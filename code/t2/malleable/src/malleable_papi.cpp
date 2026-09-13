#ifndef MALLEABLE_PAPI_CPP_INCLUDED
#define MALLEABLE_PAPI_CPP_INCLUDED

#ifdef MAL_USE_PAPI

	#include <papi.h>

#endif

#include "malleable.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef MAL_PAPI_REF_FREQ_HZ

	#define MAL_PAPI_REF_FREQ_HZ 100e6

#endif

#ifndef MAL_PAPI_STATIC_W

	#define MAL_PAPI_STATIC_W 2.0

#endif

#ifndef MAL_PAPI_DYN_NJ_PER_CYC

	#define MAL_PAPI_DYN_NJ_PER_CYC 1.0

#endif
#ifndef MAL_PAPI_MEM_NJ_PER_MISS

	#define MAL_PAPI_MEM_NJ_PER_MISS 15.0

#endif

#ifndef MAL_PAPI_MISS_PENALTY_CYCLES

	#define MAL_PAPI_MISS_PENALTY_CYCLES 200.0

#endif

#ifndef MAL_PAPI_MEM_BOUND_OOO_CORRECTION

	#define MAL_PAPI_MEM_BOUND_OOO_CORRECTION 0.7

#endif

constexpr double kRefFreqHz = MAL_PAPI_REF_FREQ_HZ;
constexpr double kStaticW = MAL_PAPI_STATIC_W;
constexpr double kDynNJPerCyc = MAL_PAPI_DYN_NJ_PER_CYC;
constexpr double kMemNJPerMiss = MAL_PAPI_MEM_NJ_PER_MISS;

double g_papi_miss_penalty_cycles = MAL_PAPI_MISS_PENALTY_CYCLES;
double g_papi_mem_bound_ooo_correction = MAL_PAPI_MEM_BOUND_OOO_CORRECTION;

inline void papi_load_env_overrides() {

	if (const char* v = std::getenv("MAL_PAPI_MISS_PENALTY_CYCLES")) {

		char* end = nullptr;
		double val = std::strtod(v, &end);

		if (end != v && val > 0.0) {

			g_papi_miss_penalty_cycles = val;
		}
	}

	if (const char* v = std::getenv("MAL_PAPI_MEM_BOUND_OOO_CORRECTION")) {

		char* end = nullptr;
		double val = std::strtod(v, &end);

		if (end != v && val >= 0.0 && val <= 1.0) {

			g_papi_mem_bound_ooo_correction = val;
		}
	}
}

#ifdef MAL_USE_PAPI

constexpr int kPapiEventCodes[kNumPapiEvents] = {

	PAPI_TOT_CYC,
	PAPI_TOT_INS,
	PAPI_L3_TCM,
	PAPI_REF_CYC,
};

int g_papi_eventset = PAPI_NULL;
bool g_papi_available = false;

void papi_init() {

	papi_load_env_overrides();

	int ret = PAPI_library_init(PAPI_VER_CURRENT);

	if (ret != PAPI_VER_CURRENT) {

		MAL_LOG(MAL_LOG_WARN, "PAPI: init failed (ret=%d) — energy metrics disabled", ret);
		return;
	}

	int es = PAPI_NULL;

	if (PAPI_create_eventset(&es) != PAPI_OK) {

		MAL_LOG(MAL_LOG_WARN, "PAPI: create_eventset failed — energy metrics disabled");
		return;
	}

	int n_added = 0;

	for (int i = 0; i < kNumPapiEvents; i++) {

		if (PAPI_add_event(es, kPapiEventCodes[i]) == PAPI_OK) {

			n_added++;
		}
	}

	if (n_added < kNumPapiEvents || PAPI_start(es) != PAPI_OK) {

		PAPI_destroy_eventset(&es);
		MAL_LOG(MAL_LOG_WARN, "PAPI: only %d/%d events added — DISABLED (a partial set would shift the value columns positionally)", n_added, kNumPapiEvents);
		return;
	}

	g_papi_eventset = es;
	g_papi_available = true;

	MAL_LOG(MAL_LOG_INFO, "PAPI: enabled (%d/%d events: TOT_CYC INS L3_TCM REF_CYC) | model: E = %.1fW×T + %.1fnJ×CYC + %.1fnJ×LLC", n_added, kNumPapiEvents, kStaticW, kDynNJPerCyc, kMemNJPerMiss);
}

bool papi_accum_epoch(long long out[kNumPapiEvents]) {

	if (!g_papi_available) {

		return false;
	}

	long long vals[kNumPapiEvents] = {};

	if (PAPI_accum(g_papi_eventset, vals) != PAPI_OK) {

		return false;
	}

	for (int i = 0; i < kNumPapiEvents; i++) {

		out[i] += vals[i];
	}

	return true;
}

void papi_finalize() {

	if (!g_papi_available) {

		return;
	}

	long long dummy[kNumPapiEvents] = {};
	PAPI_stop(g_papi_eventset, dummy);
	PAPI_destroy_eventset(&g_papi_eventset);
	PAPI_shutdown();
	g_papi_available = false;
}

bool papi_is_available() {
	return g_papi_available;
}

#else

void papi_init() {

	papi_load_env_overrides();
}

void papi_finalize() {}

bool papi_accum_epoch(long long out[kNumPapiEvents]) {

	(void)out;

	return false;
}

bool papi_is_available() {

	return false;
}

#endif

void papi_rotate_epoch(long long prev_buf[kNumPapiEvents]) {

	long long sample[kNumPapiEvents] = {};
	papi_accum_epoch(sample);
	std::copy(sample, sample + kNumPapiEvents, prev_buf);
}

double papi_energy_nj(const long long vals[kNumPapiEvents]) {

	const long long cyc = vals[0];
	const long long llc = vals[2];
	const long long refc = vals[3];

	if (cyc <= 0 && refc <= 0) {

		return 0.0;
	}

	const double t_wall_s = (refc > 0) ? (double)refc / kRefFreqHz : 0.0;
	const double e_static = kStaticW * t_wall_s * 1e9;

	const double stall_cyc = (double)std::max(0LL, llc) * g_papi_miss_penalty_cycles * g_papi_mem_bound_ooo_correction;
	const double active_cyc = std::max(0.0, (double)cyc - stall_cyc);
	const double e_dynamic = kDynNJPerCyc * active_cyc;
	const double e_memory = kMemNJPerMiss * (double)std::max(0LL, llc);

	return e_static + e_dynamic + e_memory;
}

double papi_energy_per_iter(const long long vals[kNumPapiEvents], long done) {

	if (done <= 0) {

		return 0.0;
	}

	double e = papi_energy_nj(vals);

	return (e > 0.0) ? e / (double)done : 0.0;
}

double papi_ipc(const long long vals[kNumPapiEvents]) {

	if (vals[0] <= 0 || vals[1] <= 0) {

		return 0.0;
	}

	return (double)vals[1] / (double)vals[0];
}

double papi_mem_bound_fraction(const long long vals[kNumPapiEvents]) {

	if (vals[0] <= 0) {

		return 0.0;
	}

	const double stall_cycles = (double)std::max(0LL, vals[2]) * g_papi_miss_penalty_cycles * g_papi_mem_bound_ooo_correction;

	return std::min(1.0, stall_cycles / (double)vals[0]);
}

#endif
