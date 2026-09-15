#include "malleable.hpp"
#include "example_utils.hpp"
#include "mtx_loader.hpp"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static float x_value(long c) {
	return 1.0f + (float)(c & 3) * 0.25f;
}

int main(int argc, char* argv[]) {

	mal_init();

	const long synth_n = parse_arg_long(argc, argv, "n", 100000);
	const long avg_deg = parse_arg_long(argc, argv, "avg_deg", 16);
	const char* pattern = parse_arg_str(argc, argv, "pattern", "clustered");
	const double contrast = parse_arg_double(argc, argv, "contrast", 7.0);

	CsrMatrix A;
	long M = 0, N = 0, nnz = 0;

	if (mal_rank() == 0) {

		A = build_synthetic(pattern, synth_n, avg_deg, 12345u, contrast, 0.25);
		M = A.rows;
		N = A.cols;
		nnz = A.nnz;
	}

	mal_bcast(M);
	mal_bcast(N);
	mal_bcast(nnz);

	std::vector<long> row_ptr((size_t)M + 1, 0);
	float* val = nullptr;
	int* col = nullptr;

	if (mal_rank() == 0) {

		std::memcpy(row_ptr.data(), A.row_ptr.data(), (size_t)(M + 1) * sizeof(long));

		val = static_cast<float*>(std::malloc((size_t)nnz * sizeof(float)));
		col = static_cast<int*>(std::malloc((size_t)nnz * sizeof(int)));
		std::memcpy(val, A.vals.data(), (size_t)nnz * sizeof(float));
		std::memcpy(col, A.col_idx.data(), (size_t)nnz * sizeof(int));
	}

	std::vector<float> x((size_t)N);

	for (long c = 0; c < N; c++) {

		x[(size_t)c] = x_value(c);
	}

	float* y = (mal_rank() == 0) ? static_cast<float*>(std::calloc((size_t)std::max(1L, M), sizeof(float))) : nullptr;

	if (mal_rank() == 0) {

		#if !BENCH_CSV
		MAL_LOG(MAL_LOG_INFO, "[SETUP] spmv_csr DISTRIBUTED rows=%ld cols=%ld nnz=%ld (each rank stores only its rows' nnz)", M, N, nnz);
		#endif
	}

	const double t0 = MPI_Wtime();
	long row, lim;
	MalFor f = mal_for(M, &row, &lim);

	mal_attach_csr(f, (void**)&val, sizeof(float), (void**)&col, sizeof(int), row_ptr.data(), M, nnz);
	mal_attach_vec(f, (void**)&y, sizeof(float), M, 0);

	for (; row < lim; row++) {

		double acc = 0.0;

		for (long k = row_ptr[(size_t)row]; k < row_ptr[(size_t)row + 1]; k++) {

			acc += (double)val[k] * (double)x[(size_t)col[k]];
		}

		y[row] = (float)acc;
		mal_check_for(f);
	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV
	(void)compute_seconds;
	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

		print_bench_csv("spmv_csr", "malleable", "distributed", mal_size(), mal_active_size(), M, compute_seconds, 0);

		#else

		int errors = 0;
		double max_rel = 0.0;

		for (long r = 0; r < M; r++) {

			double expected = 0.0;

			for (long k = A.row_ptr[(size_t)r]; k < A.row_ptr[(size_t)r + 1]; k++) {

				expected += (double)A.vals[(size_t)k] * (double)x_value(A.col_idx[(size_t)k]);
			}

			const double denom = std::max(1.0, std::fabs(expected));
			const double rel = std::fabs((double)y[r] - expected) / denom;

			if (rel > max_rel) {

				max_rel = rel;
			}

			if (rel > 1e-4) {

				errors++;
			}
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] spmv_csr %s (rows=%ld nnz=%ld active=%d max_rel=%.2e errors=%d)", errors == 0 ? "OK" : "WRONG", M, nnz, mal_active_size(), max_rel, errors);

		#endif
	}

	if (y) {

		std::free(y);
	}

	return EXIT_SUCCESS;
}
