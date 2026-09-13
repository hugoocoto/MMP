#include "malleable.hpp"
#include "example_utils.hpp"
#include "thread_metrics.hpp"
#include <mpi.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool rank_in_list(int r, const char* list) {

	if (!list || !*list) {

		return false;
	}

	const char* p = list;

	while (*p) {

		char* end = nullptr;
		long v = std::strtol(p, &end, 10);

		if (end == p) {

			break;
		}

		if ((long)r == v) {

			return true;
		}

		p = end;

		while (*p == ',' || *p == ' ') {

			p++;
		}
	}

	return false;
}

static void busy_wait_us(double us) {

	if (us <= 0.0) {

		return;
	}

	const auto t0 = std::chrono::steady_clock::now();
	volatile double s = 0.0;

	for (;;) {

		s += 1.0;

		const double el = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();

		if (el >= us) {

			break;
		}
	}

	(void)s;
}

static float kernel(long i, long work) {

	float acc = 0.0f;

	for (long k = 0; k < work; k++) {

		acc += std::sin((float)(i + k)) * std::cos((float)(i - k)) + std::sqrt((float)((i + 1) * (k + 1)));
	}

	return acc;
}

static float kernel_mem(long i, long work, const float* buf, size_t nbuf) {

	float acc = 0.0f;
	unsigned long long x = (unsigned long long)(i + 1) * 6364136223846793005ULL + 1442695040888963407ULL;

	for (long k = 0; k < work; k++) {

		x = x * 6364136223846793005ULL + 1442695040888963407ULL;
		acc += buf[(size_t)((x >> 33) % nbuf)];
	}

	return acc;
}

static float kernel_imbal(long i, long work, long n) {

	const long w = work / 8 + (work * 2 * i) / ((n > 0) ? n : 1);

	return kernel(i, (w > 1) ? w : 1);
}

static float kernel_dispatch(long i, long work, int kind, long n, const float* buf, size_t nbuf) {

	if (kind == 1) {

		return kernel_mem(i, work, buf, nbuf);
	}

	if (kind == 2) {

		return kernel_imbal(i, work, n);
	}

	return kernel(i, work);
}

int main(int argc, char* argv[]) {

	mal_init();

	const long N = parse_arg_long(argc, argv, "n", 20000);
	const long work = parse_arg_long(argc, argv, "work", 150);
	const long noise_us = parse_arg_long(argc, argv, "noise_us", 0);
	const char* noise_ranks = parse_arg_str(argc, argv, "noise_ranks", "");
	const char* kernel_name = parse_arg_str(argc, argv, "kernel", "fp");
	const long buf_mb = parse_arg_long(argc, argv, "buf_mb", 8);

	const bool noisy = rank_in_list(mal_rank(), noise_ranks);
	const int kernel_kind = (std::strcmp(kernel_name, "mem") == 0) ? 1 : ((std::strcmp(kernel_name, "imbal") == 0) ? 2 : 0);
	const size_t nbuf = (kernel_kind == 1) ? ((size_t)std::max(1L, buf_mb) << 20) / sizeof(float) : 1;
	float* buf = static_cast<float*>(std::malloc(nbuf * sizeof(float)));

	for (size_t k = 0; k < nbuf; k++) {

		buf[k] = (float)((k * 2654435761ULL) % 1000) * 0.001f;
	}

	float* C = (mal_rank() == 0) ? static_cast<float*>(std::calloc((size_t)std::max(1L, N), sizeof(float))) : nullptr;

	if (mal_rank() == 0) {

		#if !BENCH_CSV

		MAL_LOG(MAL_LOG_INFO, "[SETUP] noise_bench n=%ld work=%ld noise_us=%ld noise_ranks=%s", N, work, noise_us, noise_ranks[0] ? noise_ranks : "(none)");

		#endif
	}

	papi_init();
	long long papi_prime[kNumPapiEvents] = {0};
	papi_accum_epoch(papi_prime);

	PerfCounters pc = tm_perf_open();
	const std::vector<TidInfo> tids_before = tm_scan_tids();
	const double worker_cpu0 = mal_worker_cpu_seconds();
	const double worker_runq0 = mal_worker_runq_seconds();
	const ThreadSnap snap_start = tm_snapshot(pc);

	const double t0 = MPI_Wtime();
	long i, lim;
	MalFor f = mal_for(N, i, lim);

	mal_attach_vec(f, (void**)&C, sizeof(float), N, 0);

	double acc_kernel = 0.0, acc_check = 0.0;
	long iters_done = 0;
	double t_prev = tm_mono_s();

	for (; i < lim; i++) {

		C[i] = kernel_dispatch(i, work, kernel_kind, N, buf, nbuf);

		if (noisy) {

			busy_wait_us((double)noise_us);
		}

		const double t_a = tm_mono_s();
		acc_kernel += t_a - t_prev;

		mal_check_for(f);

		const double t_b = tm_mono_s();
		acc_check += t_b - t_a;
		t_prev = t_b;
		iters_done++;
	}

	const double wall_loop = MPI_Wtime() - t0;
	long long papi_vals[kNumPapiEvents] = {0};
	const bool papi_ok = papi_accum_epoch(papi_vals);
	const ThreadSnap snap_loop = tm_snapshot(pc);
	const double worker_cpu1 = mal_worker_cpu_seconds();
	const double worker_runq1 = mal_worker_runq_seconds();
	const std::vector<TidInfo> tids_after = tm_scan_tids();
	const int compute_core = tm_pinned_cpu("/proc/thread-self/status") >= 0 ? tm_pinned_cpu("/proc/thread-self/status") : tm_current_cpu();

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	tm_perf_close(pc);

	int errors = -1;

	{

		double worker_cpu = (worker_cpu0 >= 0.0 && worker_cpu1 >= 0.0) ? worker_cpu1 - worker_cpu0 : -1.0;
		double worker_runq = (worker_runq0 >= 0.0 && worker_runq1 >= 0.0) ? worker_runq1 - worker_runq0 : -1.0;
		int worker_core = mal_worker_core();
		const long self_tid = tm_self_tid();
		const long lib_wtid = mal_worker_tid();

		for (size_t k = 0; k < tids_after.size(); k++) {

			if (tids_after[k].tid == self_tid) {

				continue;
			}

			if (tids_after[k].tid != lib_wtid && tids_after[k].comm != "mal_worker") {

				continue;
			}

			double before_cpu = 0.0, before_runq = 0.0;

			for (size_t j = 0; j < tids_before.size(); j++) {

				if (tids_before[j].tid == tids_after[k].tid) {

					before_cpu = (tids_before[j].cpu_s > 0.0) ? tids_before[j].cpu_s : 0.0;
					before_runq = (tids_before[j].runq_s > 0.0) ? tids_before[j].runq_s : 0.0;
				}
			}

			if (worker_cpu < 0.0 && tids_after[k].cpu_s >= 0.0) {

				worker_cpu = tids_after[k].cpu_s - before_cpu;
			}

			if (worker_runq < 0.0 && tids_after[k].runq_s >= 0.0) {

				worker_runq = tids_after[k].runq_s - before_runq;
			}

			if (worker_core < 0) {

				worker_core = (tids_after[k].pinned_cpu >= 0) ? tids_after[k].pinned_cpu : tids_after[k].last_cpu;
			}

			break;
		}

		const double c_main = tm_delta(snap_loop.compute_cpu_s, snap_start.compute_cpu_s);
		const double p_cpu = tm_delta(snap_loop.proc_cpu_s, snap_start.proc_cpu_s);
		const double other_cpu = (c_main >= 0.0 && p_cpu >= 0.0) ? p_cpu - c_main : -1.0;

		if (mal_rank() == 0 && C) {

			errors = 0;

			for (long q = 0; q < 1024; q++) {

				const long r = (N > 0) ? ((q * 7919) % N) : 0;
				const float expected = kernel_dispatch(r, work, kernel_kind, N, buf, nbuf);
				const float denom = std::max(1.0f, std::fabs(expected));

				if (std::fabs(C[r] - expected) / denom > 1e-3f) {

					errors++;
				}
			}
		}

		const long long p_cyc = papi_ok ? papi_vals[0] : -1;
		const long long p_ins = papi_ok ? papi_vals[1] : -1;
		const long long p_l3 = papi_ok ? papi_vals[2] : -1;
		const long long p_ref = papi_ok ? papi_vals[3] : -1;
		const double p_ipc = papi_ok ? papi_ipc(papi_vals) : -1.0;
		const double p_memb = papi_ok ? papi_mem_bound_fraction(papi_vals) : -1.0;

		std::printf("TM,%d,%d,%d,%d,%d,%d,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%lld,%lld,%lld,%lld,%lld,%d,%lld,%lld,%lld,%lld,%.4f,%.4f,%d\n", mal_rank(), noisy ? 1 : 0, compute_core, worker_core, (int)tids_after.size(), tm_schedstats_enabled() ? 1 : 0, iters_done, wall_loop, compute_seconds, c_main, p_cpu, other_cpu, acc_kernel, acc_check, tm_delta(snap_loop.runq_s, snap_start.runq_s), worker_cpu, worker_runq, tm_delta_l(snap_loop.nvcsw, snap_start.nvcsw), tm_delta_l(snap_loop.nivcsw, snap_start.nivcsw), tm_delta_ll(snap_loop.cycles, snap_start.cycles), tm_delta_ll(snap_loop.insns, snap_start.insns), tm_delta_ll(snap_loop.llc_ref, snap_start.llc_ref), tm_delta_ll(snap_loop.llc_miss, snap_start.llc_miss), tm_delta_ll(snap_loop.ref_cycles, snap_start.ref_cycles), papi_ok ? 1 : 0, p_cyc, p_ins, p_l3, p_ref, p_ipc, p_memb, errors);
		std::fflush(stdout);
	}

	#if !BENCH_CSV

	(void)compute_seconds;

	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

		print_bench_csv("noise_bench", "malleable", "balanced", mal_size(), mal_active_size(), N, compute_seconds, (errors > 0) ? errors : 0);

		#else

		int errors_full = 0;
		float max_rel = 0.0f;

		for (long r = 0; r < N; r++) {

			const float expected = kernel_dispatch(r, work, kernel_kind, N, buf, nbuf);
			const float denom = std::max(1.0f, std::fabs(expected));
			const float rel = std::fabs(C[r] - expected) / denom;

			if (rel > max_rel) {

				max_rel = rel;
			}

			if (rel > 1e-3f) {

				errors++;
			}
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] noise_bench %s (n=%ld active=%d noisy=%s time=%.4f max_rel=%.2e errors=%d)", errors == 0 ? "OK" : "WRONG", N, mal_active_size(), noise_ranks[0] ? noise_ranks : "(none)", compute_seconds, max_rel, errors);

		#endif
	}

	if (C) {

		std::free(C);
	}

	if (buf) {

		std::free(buf);
	}

	papi_finalize();

	return EXIT_SUCCESS;
}
