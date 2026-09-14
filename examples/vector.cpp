#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

int main(int argc, char* argv[]) {

	mal_init();

	const long mal_n = parse_arg_long(argc, argv, "n", 200);
	const long collapse_rows = parse_arg_long(argc, argv, "rows", 4);
	const long collapse_cols = parse_arg_long(argc, argv, "cols", 50);

	bool use_collapse = (argc > 1 && std::strcmp(argv[1], "collapse") == 0);
	const long total_n = use_collapse ? (collapse_rows * collapse_cols) : mal_n;
	double compute_seconds = 0.0;

	float *A = nullptr, *B = nullptr, *C = nullptr;

	if (mal_rank() == 0) {

		A = static_cast<float*>(std::malloc(static_cast<size_t>(total_n) * sizeof(float)));
		B = static_cast<float*>(std::malloc(static_cast<size_t>(total_n) * sizeof(float)));
		C = static_cast<float*>(std::malloc(static_cast<size_t>(total_n) * sizeof(float)));

		for (long k = 0; k < total_n; k++) {

			A[k] = static_cast<float>(k + 1);
			B[k] = static_cast<float>(total_n - k);
		}
	}

	if (use_collapse) {

		long i, limit_rows, j, limit_cols;

		const long starts[2] = {0, 0};
		const long limits2d[2] = {collapse_rows, collapse_cols};

		long* iters[2] = {&i, &j};
		long* loop_limits[2] = {&limit_rows, &limit_cols};
		const double t0 = MPI_Wtime();

		MalForND nd = mal_for_nd_begin(iters, loop_limits, starts, limits2d, 2);

		mal_attach_vec(nd, (void**)&A, sizeof(float), total_n, -1);
		mal_attach_vec(nd, (void**)&B, sizeof(float), total_n, -1);
		mal_attach_vec(nd, (void**)&C, sizeof(float), total_n, 0);

		#if !BENCH_CSV

		const useconds_t delay_us = example_delay_us(200000);

		#endif

		for (; i < limit_rows; i++) {

			for (; j < limit_cols; j++) {

				long idx = i * limit_cols + j;
				float acc = 0.0f;

				for (int iter = 0; iter < 1000; iter++) {

					acc += std::sin(A[idx]) * std::cos(B[idx]) + std::sqrt(A[idx] * B[idx]);
				}

				C[idx] = acc;

				#if !BENCH_CSV

				MAL_LOG(MAL_LOG_INFO, "[ITER] C[%ld] = %.6f", idx, C[idx]);
				usleep(delay_us);

				#endif

				mal_check_for(nd);
			}
		}

		mal_finalize();
		compute_seconds = MPI_Wtime() - t0;

	} else {

		long i, limit;
		const double t0 = MPI_Wtime();

		MalFor f = mal_for(mal_n, i, limit);

		mal_attach_vec(f, (void**)&A, sizeof(float), mal_n, -1);
		mal_attach_vec(f, (void**)&B, sizeof(float), mal_n, -1);
		mal_attach_vec(f, (void**)&C, sizeof(float), mal_n, 0);

		#if !BENCH_CSV

		const useconds_t delay_us = example_delay_us(200000);

		#endif

		for (; i < limit; i++) {

			float acc = 0.0f;

			for (int iter = 0; iter < 1000; iter++) {

				acc += std::sin(A[i]) * std::cos(B[i]) + std::sqrt(A[i] * B[i]);
			}

			C[i] = acc;

			#if !BENCH_CSV

			MAL_LOG(MAL_LOG_INFO, "[ITER] C[%ld] = %.6f", i, C[i]);
			usleep(delay_us);

			#endif

			mal_check_for(f);
		}

		mal_finalize();
		compute_seconds = MPI_Wtime() - t0;
	}

	#if !BENCH_CSV

	(void)compute_seconds;

	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

		print_bench_csv("vector", "malleable", use_collapse ? "collapse" : "flat", mal_size(), mal_active_size(), total_n, compute_seconds, 0);

		#else

		int errors = 0;

		for (long k = 0; k < total_n; k++) {

			const float ak = static_cast<float>(k + 1);
			const float bk = static_cast<float>(total_n - k);
			const float expected = (std::sin(ak) * std::cos(bk) + std::sqrt(ak * bk)) * 1000.0f;

			if (std::fabs(C[k] - expected) > std::fabs(expected) * 1e-3f + 1e-3f) {

				errors++;
				break;
			}
		}

		if (errors == 0) {

			MAL_LOG(MAL_LOG_INFO, "[RESULT] vector OK");

		} else {

			MAL_LOG(MAL_LOG_ERROR, "[RESULT] vector WRONG");
		}

		#endif

		std::free(C);
	}

	return EXIT_SUCCESS;
}
