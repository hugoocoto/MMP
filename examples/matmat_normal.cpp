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

	const long M = parse_arg_long(argc, argv, "m", 120);
	const long K = parse_arg_long(argc, argv, "k", 60);
	const long N = parse_arg_long(argc, argv, "n", 40);

	if (M % world_size != 0) {

		if (world_rank == 0) {

			std::fprintf(stderr, "[ERROR] m (%ld) must be divisible by world_size (%d)\n", M, world_size);
		}

		MPI_Finalize();
		return EXIT_FAILURE;
	}

	const long local_rows = M / world_size;

	std::vector<float> full_A;
	std::vector<float> B(static_cast<size_t>(K * N));
	std::vector<float> full_C;

	if (world_rank == 0) {

		full_A.resize(static_cast<size_t>(M * K));
		full_C.resize(static_cast<size_t>(M * N));

		for (long r = 0; r < M; r++) {

			for (long c = 0; c < K; c++) {

				full_A[static_cast<size_t>(r * K + c)] = static_cast<float>(r + 1);
			}
		}

		for (long i = 0; i < K * N; i++) {

			B[static_cast<size_t>(i)] = 1.0f;
		}
	}

	std::vector<float> local_A(static_cast<size_t>(local_rows * K));
	std::vector<float> local_C(static_cast<size_t>(local_rows * N), 0.0f);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Scatter(full_A.data(), static_cast<int>(local_rows * K), MPI_FLOAT, local_A.data(), static_cast<int>(local_rows * K), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(B.data(), static_cast<int>(K * N), MPI_FLOAT, 0, MPI_COMM_WORLD);

	#if !BENCH_CSV

	const useconds_t delay_us = example_delay_us(100000);

	#endif

	for (long r = 0; r < local_rows; r++) {

		for (long j = 0; j < N; j++) {

			float acc = 0.0f;

			for (long k = 0; k < K; k++) {

				acc += local_A[static_cast<size_t>(r * K + k)] * B[static_cast<size_t>(k * N + j)];
			}

			local_C[static_cast<size_t>(r * N + j)] = acc;
		}

		#if !BENCH_CSV

		usleep(delay_us);

		#endif
	}

	MPI_Gather(local_C.data(), static_cast<int>(local_rows * N), MPI_FLOAT, full_C.data(), static_cast<int>(local_rows * N), MPI_FLOAT, 0, MPI_COMM_WORLD);

	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		#if BENCH_CSV

		print_bench_csv("matmat", "normal", "mm", world_size, world_size, M * N, t1 - t0, 0);

		#else

		int errors = 0;

		for (long r = 0; r < M && errors == 0; r++) {

			const float expected = static_cast<float>(r + 1) * static_cast<float>(K);

			for (long c = 0; c < N; c++) {

				if (std::fabs(full_C[static_cast<size_t>(r * N + c)] - expected) > 1e-3f) {

					errors++;
					break;
				}
			}
		}

		std::printf("[RESULT] mat-mat %s (M=%ld K=%ld N=%ld errors=%d)\n", errors == 0 ? "OK" : "WRONG", M, K, N, errors);
		std::printf("[TIME] matmat normal mpi np=%d seconds=%.6f\n", world_size, t1 - t0);

		#endif
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
