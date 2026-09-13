#include "example_utils.hpp"
#include "mtx_loader.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static float x_value(long c) {

	return 1.0f + (float)(c & 3) * 0.25f;
}

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);

	int rank = 0, world = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world);

	const char* file = parse_arg_str(argc, argv, "file", "");
	const long synth_n = parse_arg_long(argc, argv, "synthetic", 0);
	const long avg_deg = parse_arg_long(argc, argv, "avg_deg", 16);
	const char* pattern = parse_arg_str(argc, argv, "pattern", "clustered");
	const double contrast = parse_arg_double(argc, argv, "contrast", 7.0);
	const double cfrac = parse_arg_double(argc, argv, "cluster_frac", 0.25);
	const long n_spikes = parse_arg_long(argc, argv, "n_spikes", 4);
	const char* partition = parse_arg_str(argc, argv, "partition", "block");
	const long reps_arg = parse_arg_long(argc, argv, "reps", 50);
	const long reps = (reps_arg > 0) ? reps_arg : 1;

	CsrMatrix A;

	if (synth_n > 0) {

		A = build_synthetic(pattern, synth_n, avg_deg, 12345u, contrast, cfrac, n_spikes);

	} else if (file[0] != '\0') {

		std::string err;

		if (!load_matrix_market(file, A, err)) {

			std::fprintf(stderr, "[spmv_imbal_normal] rank %d failed to load '%s': %s\n", rank, file, err.c_str());
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

	} else {

		A = build_synthetic(pattern, 200000, avg_deg, 12345u, contrast, cfrac, n_spikes);
	}

	const long M = A.rows;

	std::vector<float> x((size_t)A.cols);

	for (long c = 0; c < A.cols; c++) {

		x[(size_t)c] = x_value(c);
	}

	if (rank == 0) {

		#if !BENCH_CSV
		std::printf("[SETUP] spmv_imbal_normal rows=%ld cols=%ld nnz=%ld max_row_nnz=%ld mean_row_nnz=%.1f reps=%ld world=%d\n", M, A.cols, A.nnz, A.max_row_nnz(), A.mean_row_nnz(), reps, world);
		#endif
	}

	const bool cost_part = (std::strcmp(partition, "cost") == 0);
	auto bounds = [&](int p) -> std::pair<long, long> {
		if (!cost_part) {

			return {(long)((double)p * (double)M / (double)world), (long)((double)(p + 1) * (double)M / (double)world)};
		}

		const double lo = (double)p * (double)A.nnz / (double)world;
		const double hi = (double)(p + 1) * (double)A.nnz / (double)world;
		long a = (long)(std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), (long)std::llround(lo)) - A.row_ptr.begin());
		long b = (p == world - 1) ? M : (long)(std::lower_bound(A.row_ptr.begin(), A.row_ptr.end(), (long)std::llround(hi)) - A.row_ptr.begin());
		a = std::min(a, M);
		b = std::min(b, M);
		b = std::max(a, b);
		return {a, b};
	};

	const auto [r0, r1] = bounds(rank);
	const long local_rows = std::max(0L, r1 - r0);
	std::vector<float> y_local((size_t)local_rows, 0.0f);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t0 = MPI_Wtime();

	for (long r = r0; r < r1; r++) {

		const long b = A.row_ptr[(size_t)r];
		const long e = A.row_ptr[(size_t)r + 1];
		double acc = 0.0;

		for (long rep = 0; rep < reps; rep++) {

			for (long j = b; j < e; j++) {

				acc += (double)A.vals[(size_t)j] * (double)x[(size_t)A.col_idx[(size_t)j]];
			}
		}

		y_local[(size_t)(r - r0)] = (float)(acc / (double)reps);
	}

	std::vector<int> counts((size_t)world, 0);
	std::vector<int> displs((size_t)world, 0);

	for (int p = 0; p < world; p++) {

		const auto [pr0, pr1] = bounds(p);
		counts[(size_t)p] = (int)(pr1 - pr0);
		displs[(size_t)p] = (int)pr0;
	}

	std::vector<float> y;

	if (rank == 0) {

		y.assign((size_t)M, 0.0f);
	}

	MPI_Gatherv(y_local.data(), (int)local_rows, MPI_FLOAT, rank == 0 ? y.data() : nullptr, counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	const double t1 = MPI_Wtime();

	if (rank == 0) {

		#if BENCH_CSV

		print_bench_csv("spmv_imbal", "normal", "spmv", world, world, M, t1 - t0, 0);

		#else

		(void)t0;
		(void)t1;
		int errors = 0;
		float max_rel = 0.0f;

		for (long r = 0; r < M; r++) {

			double expected = 0.0;

			for (long j = A.row_ptr[(size_t)r]; j < A.row_ptr[(size_t)r + 1]; j++) {

				expected += (double)A.vals[(size_t)j] * (double)x_value(A.col_idx[(size_t)j]);
			}

			const double denom = std::max(1.0, std::fabs(expected));
			const double rel = std::fabs((double)y[(size_t)r] - expected) / denom;
			max_rel = std::max(max_rel, (float)rel);

			if (rel > 1e-4) { errors++; }
		}

		std::printf("[RESULT] spmv_imbal_normal %s (rows=%ld nnz=%ld world=%d max_rel=%.2e errors=%d)\n", errors == 0 ? "OK" : "WRONG", M, A.nnz, world, max_rel, errors);

		#endif
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
