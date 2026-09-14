#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int rank = 0, world = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world);

	const long D = parse_arg_long(argc, argv, "cells", 200000);
	const long T = parse_arg_long(argc, argv, "steps", 2000);
	const double alpha = 0.25;

	const long c0 = (long)((double)rank * (double)D / (double)world);
	const long c1 = (long)((double)(rank + 1) * (double)D / (double)world);
	const long n = std::max(0L, c1 - c0);

	std::vector<double> u((size_t)n + 2, 0.0), un((size_t)n + 2, 0.0);
	auto u0 = [D](long g) { return 1.0 + std::sin(6.2831853 * (double)g / (double)D); };

	for (long i = 0; i < n; i++) {

		u[(size_t)i + 1] = u0(c0 + i);
	}

	double local_sum0 = 0.0;

	for (long i = 0; i < n; i++) {

		local_sum0 += u[(size_t)i + 1];
	}

	double sum0 = 0.0;
	MPI_Allreduce(&local_sum0, &sum0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	const int left = (rank - 1 + world) % world;
	const int right = (rank + 1) % world;

	int reduces = 1;

	if (const char* v = std::getenv("MAL_STENCIL_RESID_REDUCES")) {

		long r = std::strtol(v, nullptr, 10);

		if (r > 0) {

			reduces = (int)r;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	for (long t = 0; t < T; t++) {

		double sendL = u[1], sendR = u[(size_t)n], recvL = 0.0, recvR = 0.0;
		MPI_Sendrecv(&sendL, 1, MPI_DOUBLE, left, 0, &recvR, 1, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&sendR, 1, MPI_DOUBLE, right, 1, &recvL, 1, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		u[0] = recvL;
		u[(size_t)n + 1] = recvR;

		double local_res = 0.0;

		for (long i = 1; i <= n; i++) {

			un[(size_t)i] = u[(size_t)i] + alpha * (u[(size_t)i - 1] - 2.0 * u[(size_t)i] + u[(size_t)i + 1]);
			const double d = un[(size_t)i] - u[(size_t)i];
			local_res += d * d;
		}

		for (int rr = 0; rr < reduces; rr++) {

			double res = 0.0;
			MPI_Allreduce(&local_res, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			local_res = res / (double)world;
		}

		u.swap(un);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	const double t1 = MPI_Wtime();

	double local_sum = 0.0;

	for (long i = 0; i < n; i++) {

		local_sum += u[(size_t)i + 1];
	}

	double sum = 0.0;
	MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	if (rank == 0) {

		#if BENCH_CSV

		print_bench_csv("heat", "normal", "stencil", world, world, D, t1 - t0, 0);

		#else

		(void)t0;
		(void)t1;
		const double rel = std::fabs(sum - sum0) / std::max(1.0, std::fabs(sum0));
		std::printf("[RESULT] heat_normal %s (cells=%ld steps=%ld world=%d sum_drift=%.2e)\n", rel < 1e-6 ? "OK" : "WRONG", D, T, world, rel);

		#endif
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
