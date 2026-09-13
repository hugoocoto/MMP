#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	const long mal_n = parse_arg_long(argc, argv, "n", 200);
	const long collapse_rows = parse_arg_long(argc, argv, "rows", 4);
	const long collapse_cols = parse_arg_long(argc, argv, "cols", 50);
	const bool use_collapse = (argc > 1 && std::strcmp(argv[1], "collapse") == 0);
	const long total_n = use_collapse ? (collapse_rows * collapse_cols) : mal_n;

	if (total_n % world_size != 0) {

		if (world_rank == 0) {

			std::fprintf(stderr, "[ERROR] n (%ld) must be divisible by world_size (%d)\n", total_n, world_size);
		}

		MPI_Finalize();

		return EXIT_FAILURE;
	}

	const long local_count = total_n / world_size;

	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;

	if (world_rank == 0) {

		A = static_cast<float*>(std::malloc(static_cast<size_t>(total_n) * sizeof(float)));
		B = static_cast<float*>(std::malloc(static_cast<size_t>(total_n) * sizeof(float)));
		C = static_cast<float*>(std::malloc(static_cast<size_t>(total_n) * sizeof(float)));

		for (long k = 0; k < total_n; k++) {

			A[k] = static_cast<float>(k + 1);
			B[k] = static_cast<float>(total_n - k);
		}
	}

	std::vector<float> local_a(static_cast<size_t>(local_count));
	std::vector<float> local_b(static_cast<size_t>(local_count));
	std::vector<float> local_c(static_cast<size_t>(local_count));

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Scatter(A, static_cast<int>(local_count), MPI_FLOAT, local_a.data(), static_cast<int>(local_count), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, static_cast<int>(local_count), MPI_FLOAT, local_b.data(), static_cast<int>(local_count), MPI_FLOAT, 0, MPI_COMM_WORLD);

	#if !BENCH_CSV

	const useconds_t delay_us = example_delay_us(200000);

	#endif

	for (long local_i = 0; local_i < local_count; local_i++) {

		float acc = 0.0f;

		for (int iter = 0; iter < 1000; iter++) {

			acc += std::sin(local_a[local_i]) * std::cos(local_b[local_i]) + std::sqrt(local_a[local_i] * local_b[local_i]);
		}

		local_c[local_i] = acc;

		#if !BENCH_CSV

		usleep(delay_us);

		#endif
	}

	MPI_Gather(local_c.data(), static_cast<int>(local_count), MPI_FLOAT, C, static_cast<int>(local_count), MPI_FLOAT, 0, MPI_COMM_WORLD);

	const double t1 = MPI_Wtime();
	const double elapsed = t1 - t0;

	if (world_rank == 0) {

		int errors = 0;

		#if BENCH_CSV

		print_bench_csv("vector", "normal", use_collapse ? "collapse" : "flat", world_size, world_size, total_n, elapsed, errors);

		#else

		for (long k = 0; k < total_n; k++) {

			const float expected = (std::sin(A[k]) * std::cos(B[k]) + std::sqrt(A[k] * B[k])) * 1000.0f;

			if (std::abs(C[k] - expected) > 1e-3f) {

				errors++;
			}
		}

		if (errors == 0) {

			std::printf("[RESULT] vector OK\n");

		} else {

			std::printf("[RESULT] vector WRONG\n");
		}

		std::printf("[TIME] vector normal mpi mode=%s np=%d seconds=%.6f\n", use_collapse ? "collapse" : "flat", world_size, elapsed);

		#endif

		std::free(A);
		std::free(B);
		std::free(C);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
