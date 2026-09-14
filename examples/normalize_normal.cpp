#include "example_utils.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int rank = 0, size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	const long N = parse_arg_long(argc, argv, "n", 4000);

	const long base = (size > 0) ? N / size : 0;
	const long rem = (size > 0) ? N % size : 0;

	std::vector<int> counts((size_t)size), displs((size_t)size);
	long off = 0;

	for (int r = 0; r < size; r++) {

		const long c = base + (r < rem ? 1 : 0);
		counts[(size_t)r] = (int)c;
		displs[(size_t)r] = (int)off;
		off += c;
	}

	const long local_count = counts[(size_t)rank];

	double* x = (rank == 0) ? static_cast<double*>(std::malloc((size_t)std::max(1L, N) * sizeof(double))) : nullptr;
	double* y = (rank == 0) ? static_cast<double*>(std::malloc((size_t)std::max(1L, N) * sizeof(double))) : nullptr;

	if (rank == 0) {

		for (long g = 0; g < N; g++) {

			x[g] = 1.0 + (double)(g % 10);
		}
	}

	std::vector<double> lx((size_t)std::max(1L, local_count));
	std::vector<double> ly((size_t)std::max(1L, local_count));

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	MPI_Scatterv(x, counts.data(), displs.data(), MPI_DOUBLE, lx.data(), (int)local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double local_ss = 0.0;

	for (long i = 0; i < local_count; i++) {

		local_ss += lx[(size_t)i] * lx[(size_t)i];
	}

	double sum_sq = 0.0;
	MPI_Allreduce(&local_ss, &sum_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	const double norm = std::sqrt(sum_sq);

	for (long i = 0; i < local_count; i++) {

		ly[(size_t)i] = (norm > 0.0) ? lx[(size_t)i] / norm : 0.0;
	}

	MPI_Gatherv(ly.data(), (int)local_count, MPI_DOUBLE, y, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	const double elapsed = MPI_Wtime() - t0;

	if (rank == 0) {

		#if BENCH_CSV

		print_bench_csv("normalize", "normal", "twoloop", size, size, N, elapsed, 0);

		#else

		(void)elapsed;

		int errors = 0;
		double out_sumsq = 0.0;

		for (long g = 0; g < N; g++) {

			const double expected = (norm > 0.0) ? x[g] / norm : 0.0;

			if (std::fabs(y[g] - expected) > 1e-9 * std::fabs(expected) + 1e-12) {

				errors++;
			}

			out_sumsq += y[g] * y[g];
		}

		const double unit_drift = (N > 0) ? std::fabs(std::sqrt(out_sumsq) - 1.0) : 0.0;
		const double unit_tol = 1e-12 * (double)N + 1e-9;

		std::printf("[RESULT] normalize %s (n=%ld active=%d norm=%.6f errors=%d unit_drift=%.2e)\n", (errors == 0 && unit_drift < unit_tol) ? "OK" : "WRONG", N, size, norm, errors, unit_drift);

		#endif

		std::free(x);
		std::free(y);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
