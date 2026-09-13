#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char* argv[]) {

	mal_init(MAL_RESIZE_POLICY_AUTO);

	const long M = parse_arg_long(argc, argv, "rows", 1000);
	const long K = parse_arg_long(argc, argv, "inner", 1000);
	const long N = parse_arg_long(argc, argv, "cols", 1000);
	const long nnz_a = parse_arg_long(argc, argv, "nnz-a", 500);
	const long nnz_b = parse_arg_long(argc, argv, "nnz-b", 500);

	float *A = nullptr, *B = nullptr, *C = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<float*>(std::calloc(static_cast<size_t>(M * K), sizeof(float)));
		B = static_cast<float*>(std::calloc(static_cast<size_t>(K * N), sizeof(float)));
		C = static_cast<float*>(std::calloc(static_cast<size_t>(M * N), sizeof(float)));

		const long nnz_a_row = std::min(nnz_a, K);

		for (long r = 0; r < M; r++) {

			for (long k = 0; k < nnz_a_row; k++) {

				const long col = (r * 13 + k * 7) % K;
				A[r * K + col] = 0.5f + static_cast<float>((r + k) % 11) * 0.2f;
			}
		}

		const long nnz_b_row = std::min(nnz_b, N);

		for (long r = 0; r < K; r++) {

			for (long k = 0; k < nnz_b_row; k++) {

				const long col = (r * 17 + k * 5) % N;
				B[r * N + col] = 0.3f + static_cast<float>((r + k) % 7) * 0.15f;
			}
		}

		#if !BENCH_CSV

		MAL_LOG(MAL_LOG_INFO, "[SETUP] sparse rows=%ld inner=%ld cols=%ld nnz-a=%ld nnz-b=%ld mode=auto", M, K, N, nnz_a, nnz_b);

		#endif
	}

	#if !BENCH_CSV

	float* A_ref = nullptr;
	float* B_ref = nullptr;

	if (mal_rank() == 0) {

		A_ref = static_cast<float*>(std::malloc(static_cast<size_t>(M * K) * sizeof(float)));
		B_ref = static_cast<float*>(std::malloc(static_cast<size_t>(K * N) * sizeof(float)));

		std::memcpy(A_ref, A, static_cast<size_t>(M * K) * sizeof(float));
		std::memcpy(B_ref, B, static_cast<size_t>(K * N) * sizeof(float));
	}

	#endif

	const double t0 = MPI_Wtime();
	long row, limit;
	MalFor f = mal_for(M, row, limit);

	#if !BENCH_CSV

	const useconds_t delay_us = example_delay_us(200000);

	#endif

	mal_attach_mat(f, (void**)&A, sizeof(float), M, K, -1);
	mal_attach_mat(f, (void**)&B, sizeof(float), K, N, -1, MAL_ATTACH_SHARED_ACTIVE, MAL_ATTACH_INHERIT, MAL_ACCESS_READ_ONLY);
	mal_attach_mat(f, (void**)&C, sizeof(float), M, N, 0);

	for (; row < limit; row++) {

		std::memset(&C[row * N], 0, static_cast<size_t>(N) * sizeof(float));

		for (long i = 0; i < K; i++) {

			const float a_val = A[row * K + i];

			if (a_val == 0.0f) {

				continue;
			}

			for (long c = 0; c < N; c++) {

				C[row * N + c] += a_val * B[i * N + c];
			}
		}

		#if !BENCH_CSV

		MAL_LOG(MAL_LOG_INFO, "[ITER] row=%ld", row);
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

		print_bench_csv("sparse", "malleable", "std", mal_size(), mal_active_size(), M, compute_seconds, 0);

		#else

		int errors = 0;
		float max_err = 0.0f;

		for (long r = 0; r < M && errors == 0; r++) {

			for (long c = 0; c < N; c++) {

				float expected = 0.0f;

				for (long i = 0; i < K; i++) {

					expected += A_ref[r * K + i] * B_ref[i * N + c];
				}

				const float err = std::fabs(C[r * N + c] - expected);
				max_err = std::max(max_err, err);

				if (err > 1e-3f) {

					errors++;
					break;
				}
			}
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] sparse mat-mat %s (rows=%ld inner=%ld cols=%ld errors=%d max_err=%.3e)", errors == 0 ? "OK" : "WRONG", M, K, N, errors, max_err);

		#endif
	}

	#if !BENCH_CSV

	if (A_ref) {

		std::free(A_ref);
	}

	if (B_ref) {

		std::free(B_ref);
	}

	#endif

	if (A) {

		std::free(A);
	}

	if (B) {

		std::free(B);
	}

	if (C) {

		std::free(C);
	}

	return EXIT_SUCCESS;
}
