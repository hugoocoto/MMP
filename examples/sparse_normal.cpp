#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	const long M = parse_arg_long(argc, argv, "rows", 1000);
	const long K = parse_arg_long(argc, argv, "inner", 1000);
	const long N = parse_arg_long(argc, argv, "cols", 1000);

	const long nnz_a = parse_arg_long(argc, argv, "nnz-a", 500);
	const long nnz_b = parse_arg_long(argc, argv, "nnz-b", 500);

	if (M % world_size != 0) {

		if (world_rank == 0) {

			std::fprintf(stderr, "[ERROR] rows (%ld) must be divisible by world_size (%d)\n", M, world_size);
		}

		MPI_Finalize();
		return EXIT_FAILURE;
	}

	const long local_rows = M / world_size;

	std::vector<float> full_A;
	std::vector<float> B(static_cast<size_t>(K * N), 0.0f);
	std::vector<float> full_C;

	if (world_rank == 0) {

		full_A.resize(static_cast<size_t>(M * K), 0.0f);
		full_C.resize(static_cast<size_t>(M * N), 0.0f);

		const long nnz_a_row = std::min(nnz_a, K);

		for (long r = 0; r < M; r++) {

			for (long k = 0; k < nnz_a_row; k++) {

				const long col = (r * 13 + k * 7) % K;
				full_A[static_cast<size_t>(r * K + col)] = 0.5f + static_cast<float>((r + k) % 11) * 0.2f;
			}
		}

		const long nnz_b_row = std::min(nnz_b, N);

		for (long r = 0; r < K; r++) {

			for (long k = 0; k < nnz_b_row; k++) {

				const long col = (r * 17 + k * 5) % N;
				B[static_cast<size_t>(r * N + col)] = 0.3f + static_cast<float>((r + k) % 7) * 0.15f;
			}
		}
	}

	std::vector<float> local_A(static_cast<size_t>(local_rows * K));
	std::vector<float> local_C(static_cast<size_t>(local_rows * N), 0.0f);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Scatter(full_A.data(), static_cast<int>(local_rows * K), MPI_FLOAT, local_A.data(), static_cast<int>(local_rows * K), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(B.data(), static_cast<int>(K * N), MPI_FLOAT, 0, MPI_COMM_WORLD);

	#if !BENCH_CSV

	const useconds_t delay_us = example_delay_us(200000);

	#endif

	for (long r = 0; r < local_rows; r++) {

		for (long i = 0; i < K; i++) {

			const float a_val = local_A[static_cast<size_t>(r * K + i)];

			if (a_val == 0.0f) {

				continue;
			}

			for (long c = 0; c < N; c++) {

				local_C[static_cast<size_t>(r * N + c)] += a_val * B[static_cast<size_t>(i * N + c)];
			}
		}

		#if !BENCH_CSV

		usleep(delay_us);

		#endif
	}

	MPI_Gather(local_C.data(), static_cast<int>(local_rows * N), MPI_FLOAT, full_C.data(), static_cast<int>(local_rows * N), MPI_FLOAT, 0, MPI_COMM_WORLD);

	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		#if BENCH_CSV

		print_bench_csv("sparse", "normal", "std", world_size, world_size, M, t1 - t0, 0);

		#else

		int errors = 0;
		float max_err = 0.0f;

		for (long r = 0; r < M && errors == 0; r++) {

			for (long c = 0; c < N; c++) {

				float expected = 0.0f;

				for (long i = 0; i < K; i++) {

					expected += full_A[static_cast<size_t>(r * K + i)] * B[static_cast<size_t>(i * N + c)];
				}

				const float err = std::fabs(full_C[static_cast<size_t>(r * N + c)] - expected);
				max_err = std::max(max_err, err);

				if (err > 1e-3f) {

					errors++;
					break;
				}
			}
		}

		std::printf("[RESULT] sparse mat-mat %s (M=%ld K=%ld N=%ld errors=%d max_err=%.3e)\n", errors == 0 ? "OK" : "WRONG", M, K, N, errors, max_err);
		std::printf("[TIME] sparse normal mpi np=%d seconds=%.6f\n", world_size, t1 - t0);

		#endif
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
