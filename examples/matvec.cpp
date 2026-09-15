#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {

	mal_init();

	const long M = parse_arg_long(argc, argv, "m", 1000);
	const long K = parse_arg_long(argc, argv, "k", 1000);
	const long inner_reps_arg = parse_arg_long(argc, argv, "inner", 1000);
	const long inner_reps = (inner_reps_arg > 0) ? inner_reps_arg : 1;

	float *A = nullptr, *x = nullptr, *y = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<float*>(std::malloc(static_cast<size_t>(M * K) * sizeof(float)));
		x = static_cast<float*>(std::malloc(static_cast<size_t>(K) * sizeof(float)));
		y = static_cast<float*>(std::malloc(static_cast<size_t>(M) * sizeof(float)));

		for (long r = 0; r < M; r++) {

			for (long c = 0; c < K; c++) {

				A[r * K + c] = static_cast<float>(r + 1);
			}
		}

		for (long c = 0; c < K; c++) {

			x[c] = 1.0f;
		}
	}

	const double t0 = MPI_Wtime();
	long i, lim;
	MalFor f = mal_for(M, &i, &lim);

	#if !BENCH_CSV

	const useconds_t delay_us = example_delay_us(100000);

	#endif

	mal_attach_mat(f, (void**)&A, sizeof(float), M, K, -1);
	mal_attach_vec(f, (void**)&x, sizeof(float), K, -1, MAL_ATTACH_SHARED_ALL, MAL_ATTACH_INHERIT, MAL_ACCESS_READ_ONLY);
	mal_attach_vec(f, (void**)&y, sizeof(float), M, 0);

	for (; i < lim; i++) {

		float acc = 0.0f;

		for (long iter = 0; iter < inner_reps; iter++) {

			acc = 0.0f;

			for (long k = 0; k < K; k++) {

				acc += A[i * K + k] * x[k];
			}
		}

		y[i] = acc;

		#if !BENCH_CSV

		MAL_LOG(MAL_LOG_INFO, "[MV] y[%ld] = %.1f", i, acc);
		usleep(delay_us);

		#endif

		mal_check_for(f);
	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV

	(void)compute_seconds;

	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

		print_bench_csv("matvec", "malleable", "mv", mal_size(), mal_active_size(), M, compute_seconds, 0);

		#else

		int errors = 0;

		for (long r = 0; r < M; r++) {

			const float expected = static_cast<float>(r + 1) * static_cast<float>(K);

			if (std::fabs(y[r] - expected) > 1e-3f) {

				errors = 1;
				break;
			}
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] mat-vec %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);

		#endif

		std::free(y);
	}

	std::free(x);

	return EXIT_SUCCESS;
}
