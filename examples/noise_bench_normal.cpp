#include "example_utils.hpp"
#include "thread_metrics.hpp"
#include <mpi.h>
#include <algorithm>
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

	MPI_Init(&argc, &argv);

	int rank = 0, size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const long N = parse_arg_long(argc, argv, "n", 20000);
	const long work = parse_arg_long(argc, argv, "work", 150);
	const long noise_us = parse_arg_long(argc, argv, "noise_us", 0);
	const char* noise_ranks = parse_arg_str(argc, argv, "noise_ranks", "");
	const char* kernel_name = parse_arg_str(argc, argv, "kernel", "fp");
	const long buf_mb = parse_arg_long(argc, argv, "buf_mb", 8);

	const bool noisy = rank_in_list(rank, noise_ranks);
	const int kernel_kind = (std::strcmp(kernel_name, "mem") == 0) ? 1 : ((std::strcmp(kernel_name, "imbal") == 0) ? 2 : 0);
	const size_t nbuf = (kernel_kind == 1) ? ((size_t)std::max(1L, buf_mb) << 20) / sizeof(float) : 1;
	float* kbuf = static_cast<float*>(std::malloc(nbuf * sizeof(float)));

	for (size_t k = 0; k < nbuf; k++) {

		kbuf[k] = (float)((k * 2654435761ULL) % 1000) * 0.001f;
	}

	const long base = (size > 0) ? N / size : 0;
	const long rem = (size > 0) ? N % size : 0;

	std::vector<int> counts((size_t)size), displs((size_t)size);
	long off = 0;

	for (int r = 0; r < size; r++) {

		const long c = base + (r < rem ? 1 : 0);
		counts[(size_t)r] = (int)c;
		displs[(size_t)r] = (int)off;
		off += c;
	}

	const long local_count = counts[(size_t)rank];
	const long my_start = displs[(size_t)rank];

	float* C = (rank == 0) ? static_cast<float*>(std::calloc((size_t)std::max(1L, N), sizeof(float))) : nullptr;
	std::vector<float> lc((size_t)std::max(1L, local_count));

	PerfCounters pc = tm_perf_open();
	const ThreadSnap snap_start = tm_snapshot(pc);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	double acc_kernel = 0.0;
	long iters_done = 0;
	double t_prev = tm_mono_s();

	for (long i = 0; i < local_count; i++) {

		lc[(size_t)i] = kernel_dispatch(my_start + i, work, kernel_kind, N, kbuf, nbuf);

		if (noisy) {

			busy_wait_us((double)noise_us);
		}

		const double t_a = tm_mono_s();
		acc_kernel += t_a - t_prev;
		t_prev = t_a;
		iters_done++;
	}

	const double wall_loop = MPI_Wtime() - t0;
	const ThreadSnap snap_loop = tm_snapshot(pc);
	const int compute_core = (tm_pinned_cpu("/proc/thread-self/status") >= 0) ? tm_pinned_cpu("/proc/thread-self/status") : tm_current_cpu();

	MPI_Gatherv(lc.data(), (int)local_count, MPI_FLOAT, C, counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

	const double compute_seconds = MPI_Wtime() - t0;

	tm_perf_close(pc);

	int errors = -1;

	{

		if (rank == 0 && C) {

			errors = 0;

			for (long q = 0; q < 1024; q++) {

				const long r = (N > 0) ? ((q * 7919) % N) : 0;
				const float expected = kernel_dispatch(r, work, kernel_kind, N, kbuf, nbuf);
				const float denom = std::max(1.0f, std::fabs(expected));

				if (std::fabs(C[r] - expected) / denom > 1e-3f) {

					errors++;
				}
			}
		}

		const double c_main = tm_delta(snap_loop.compute_cpu_s, snap_start.compute_cpu_s);
		const double p_cpu = tm_delta(snap_loop.proc_cpu_s, snap_start.proc_cpu_s);
		const double other_cpu = (c_main >= 0.0 && p_cpu >= 0.0) ? p_cpu - c_main : -1.0;

		std::printf("TM,%d,%d,%d,%d,%d,%d,%ld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%lld,%lld,%lld,%lld,%lld,%d\n", rank, noisy ? 1 : 0, compute_core, -1, 1, tm_schedstats_enabled() ? 1 : 0, iters_done, wall_loop, compute_seconds, c_main, p_cpu, other_cpu, acc_kernel, 0.0, tm_delta(snap_loop.runq_s, snap_start.runq_s), -1.0, -1.0, tm_delta_l(snap_loop.nvcsw, snap_start.nvcsw), tm_delta_l(snap_loop.nivcsw, snap_start.nivcsw), tm_delta_ll(snap_loop.cycles, snap_start.cycles), tm_delta_ll(snap_loop.insns, snap_start.insns), tm_delta_ll(snap_loop.llc_ref, snap_start.llc_ref), tm_delta_ll(snap_loop.llc_miss, snap_start.llc_miss), tm_delta_ll(snap_loop.ref_cycles, snap_start.ref_cycles), errors);
		std::fflush(stdout);
	}

	if (rank == 0) {

		#if BENCH_CSV

		print_bench_csv("noise_bench", "normal", "balanced", size, size, N, compute_seconds, (errors > 0) ? errors : 0);

		#else

		(void)compute_seconds;

		int errors_full = 0;
		float max_rel = 0.0f;

		for (long r = 0; r < N; r++) {

			const float expected = kernel_dispatch(r, work, kernel_kind, N, kbuf, nbuf);
			const float denom = std::max(1.0f, std::fabs(expected));
			const float rel = std::fabs(C[r] - expected) / denom;

			if (rel > max_rel) {

				max_rel = rel;
			}

			if (rel > 1e-3f) {

				errors++;
			}
		}

		std::printf("[RESULT] noise_bench_normal %s (n=%ld active=%d noisy=%s time=%.4f max_rel=%.2e errors=%d)\n", errors == 0 ? "OK" : "WRONG", N, size, noise_ranks[0] ? noise_ranks : "(none)", compute_seconds, max_rel, errors);

		#endif
	}

	if (C) {

		std::free(C);
	}

	if (kbuf) {

		std::free(kbuf);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
