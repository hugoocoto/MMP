#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[]) {

	mal_init();

	const long N = parse_arg_long(argc, argv, "n", 4000);

	std::vector<double> x((size_t)N);

	for (long g = 0; g < N; g++) {

		x[(size_t)g] = 1.0 + (double)(g % 10);
	}

	double* y = (mal_rank() == 0) ? static_cast<double*>(std::malloc((size_t)std::max(1L, N) * sizeof(double))) : nullptr;

	const double t0 = MPI_Wtime();

	double sum_sq = 0.0;
	long i1, lim1;
	MalFor f1 = mal_for(N, &i1, &lim1);

	mal_attach_acc(f1, &sum_sq);

	for (; i1 < lim1; i1++) {

		sum_sq += x[(size_t)i1] * x[(size_t)i1];
		mal_check_for(f1);
	}

	mal_sync(f1, &sum_sq, 1);
	const double norm = std::sqrt(sum_sq);

	long i2, lim2;
	MalFor f2 = mal_for(N, &i2, &lim2);

	mal_attach_vec(f2, (void**)&y, sizeof(double), N, 0);

	for (; i2 < lim2; i2++) {

		y[i2] = (norm > 0.0) ? x[(size_t)i2] / norm : 0.0;
		mal_check_for(f2);
	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV

	(void)compute_seconds;

	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

		print_bench_csv("normalize", "malleable", "twoloop", mal_size(), mal_active_size(), N, compute_seconds, 0);

		#else

		double ref_sumsq = 0.0;

		for (long g = 0; g < N; g++) {

			ref_sumsq += x[(size_t)g] * x[(size_t)g];
		}

		const double ref_norm = std::sqrt(ref_sumsq);

		int errors = 0;
		double out_sumsq = 0.0;

		for (long g = 0; g < N; g++) {

			const double expected = (ref_norm > 0.0) ? x[(size_t)g] / ref_norm : 0.0;

			if (std::fabs(y[g] - expected) > 1e-9 * std::fabs(expected) + 1e-12) {

				errors++;
			}

			out_sumsq += y[g] * y[g];
		}

		const double unit_drift = (N > 0) ? std::fabs(std::sqrt(out_sumsq) - 1.0) : 0.0;
		const double unit_tol = 1e-12 * (double)N + 1e-9;

		MAL_LOG(MAL_LOG_INFO, "[RESULT] normalize %s (n=%ld active=%d norm=%.6f errors=%d unit_drift=%.2e)", (errors == 0 && unit_drift < unit_tol) ? "OK" : "WRONG", N, mal_active_size(), norm, errors, unit_drift);

		#endif

		std::free(y);
	}

	return EXIT_SUCCESS;
}
