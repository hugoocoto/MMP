#include "example_utils.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

struct BlockRange {

	long start;
	long count;
};

BlockRange block_range(long total, int rank, int size) {

	const long base = total / size;
	const long rem = total % size;
	const long count = base + (rank < rem ? 1 : 0);
	const long start = rank * base + std::min<long>(rank, rem);

	return {start, count};
}

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int world_rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	const long total_points = parse_arg_long(argc, argv, "n", 20);
	unsigned int seed = static_cast<unsigned int>(0);
	const BlockRange range = block_range(total_points, world_rank, world_size);

	#if !BENCH_CSV

	const useconds_t delay_us = example_delay_us(200000);

	#endif

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	long local_hits = 0;

	for (long i = 0; i < range.count; i++) {

		double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
		double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

		if (x * x + y * y <= 1.0) {

			local_hits++;
		}

		#if !BENCH_CSV

		usleep(delay_us);

		#endif
	}

	long hits = 0;
	MPI_Reduce(&local_hits, &hits, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	const double t1 = MPI_Wtime();

	if (world_rank == 0) {

		#if BENCH_CSV

		print_bench_csv("montecarlo", "normal", "std", world_size, world_size, total_points, t1 - t0, 0);

		#else

		const double pi_approx = 4.0 * static_cast<double>(hits) / static_cast<double>(total_points);

		std::printf("[RESULT] montecarlo OK total_points=%ld hits=%ld pi~=%.6f error=%.2e\n", total_points, hits, pi_approx, std::fabs(pi_approx - 3.14159265358979));
		std::printf("[TIME] montecarlo normal mpi np=%d seconds=%.6f\n", world_size, t1 - t0);

		#endif
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
