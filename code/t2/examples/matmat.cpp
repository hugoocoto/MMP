#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {

	mal_init();

	const long M = parse_arg_long(argc, argv, "m", 120);
	const long K = parse_arg_long(argc, argv, "k", 60);
	const long N = parse_arg_long(argc, argv, "n", 40);

	float *A = nullptr, *B = nullptr, *C = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<float*>(std::malloc(static_cast<size_t>(M * K) * sizeof(float)));
		B = static_cast<float*>(std::malloc(static_cast<size_t>(K * N) * sizeof(float)));
		C = static_cast<float*>(std::malloc(static_cast<size_t>(M * N) * sizeof(float)));

		for (long r = 0; r < M; r++) {

			for (long c = 0; c < K; c++) {

				A[r * K + c] = static_cast<float>(r + 1);
			}
		}

		for (long i = 0; i < K * N; i++) {

			B[i] = 1.0f;
		}
	}

	const double t0 = MPI_Wtime();
	long i, lim;
	MalFor f = mal_for(M, i, lim);

	#if !BENCH_CSV

	const useconds_t delay_us = example_delay_us(100000);

	#endif

	mal_attach_mat(f, (void**)&A, sizeof(float), M, K, -1);
	mal_attach_mat(f, (void**)&B, sizeof(float), K, N, -1, MAL_ATTACH_SHARED_ALL, MAL_ATTACH_INHERIT, MAL_ACCESS_READ_ONLY);
	mal_attach_mat(f, (void**)&C, sizeof(float), M, N, 0);

	for (; i < lim; i++) {

		for (long j = 0; j < N; j++) {

			float acc = 0.0f;

			for (long k = 0; k < K; k++) {

				acc += A[i * K + k] * B[k * N + j];
			}

			C[i * N + j] = acc;
		}

		#if !BENCH_CSV

		MAL_LOG(MAL_LOG_INFO, "[MM] C[%ld, 0..%ld] computed, C[%ld,0]=%.1f", i, N - 1, i, C[i * N]);
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

		print_bench_csv("matmat", "malleable", "mm", mal_size(), mal_active_size(), M * N, compute_seconds, 0);

		#else

		int errors = 0;

		for (long r = 0; r < M && errors == 0; r++) {

			const float expected = static_cast<float>(r + 1) * static_cast<float>(K);

			for (long c = 0; c < N; c++) {

				if (std::fabs(C[r * N + c] - expected) > 1e-3f) {

					errors = 1;
					break;
				}
			}
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] mat-mat %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);

		#endif

		std::free(C);
	}

	std::free(B);

	return EXIT_SUCCESS;
}
