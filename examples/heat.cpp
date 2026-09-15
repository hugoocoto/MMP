#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[]) {

	mal_init();

	const long D = parse_arg_long(argc, argv, "cells", 200000);
	const long T = parse_arg_long(argc, argv, "steps", 2000);
	const double alpha = 0.25;

	std::vector<double> u((size_t)D), un((size_t)D);

	for (long g = 0; g < D; g++) {

		u[(size_t)g] = 1.0 + std::sin(6.2831853 * (double)g / (double)D);
	}

	un = u;

	double sum0 = 0.0;

	for (long g = 0; g < D; g++) {

		sum0 += u[(size_t)g];
	}

	if (mal_rank() == 0) {

		#if !BENCH_CSV
		MAL_LOG(MAL_LOG_INFO, "[SETUP] heat cells=%ld steps=%ld universe=%d", D, T, mal_size());
		#endif
	}

	const double t0 = MPI_Wtime();

	for (long t = 0; t < T; t++) {

		mal_loop_horizon(T - t);

		long i, lim;
		MalFor f = mal_for(D, &i, &lim);

		for (; i < lim; i++) {

			const long im = (i == 0) ? D - 1 : i - 1;
			const long ip = (i == D - 1) ? 0 : i + 1;
			un[(size_t)i] = u[(size_t)i] + alpha * (u[(size_t)im] - 2.0 * u[(size_t)i] + u[(size_t)ip]);
		}

		mal_step(f, un.data(), sizeof(double), D);

		u.swap(un);
	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV
	(void)compute_seconds;
	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

		print_bench_csv("heat", "malleable", "stencil", mal_size(), mal_active_size(), D, compute_seconds, 0);

		#else

		double sum = 0.0;

		for (long g = 0; g < D; g++) {

			sum += u[(size_t)g];
		}

		const double rel = std::fabs(sum - sum0) / std::max(1.0, std::fabs(sum0));
		MAL_LOG(MAL_LOG_INFO, "[RESULT] heat %s (cells=%ld steps=%ld active=%d sum_drift=%.2e)", rel < 1e-6 ? "OK" : "WRONG", D, T, mal_active_size(), rel);

		#endif
	}

	return EXIT_SUCCESS;
}
