#include "malleable.hpp"
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

	mal_init();

	const char* file = parse_arg_str(argc, argv, "file", "");
	const long synth_n = parse_arg_long(argc, argv, "synthetic", 0);
	const long avg_deg = parse_arg_long(argc, argv, "avg_deg", 16);
	const char* pattern = parse_arg_str(argc, argv, "pattern", "clustered");
	const double contrast = parse_arg_double(argc, argv, "contrast", 7.0);
	const double cfrac = parse_arg_double(argc, argv, "cluster_frac", 0.25);
	const long n_spikes = parse_arg_long(argc, argv, "n_spikes", 4);
	const long reps_arg = parse_arg_long(argc, argv, "reps", 50);
	const long reps = (reps_arg > 0) ? reps_arg : 1;

	CsrMatrix A;

	if (synth_n > 0) {

		A = build_synthetic(pattern, synth_n, avg_deg, 12345u, contrast, cfrac, n_spikes);

	} else if (file[0] != '\0') {

		std::string err;

		if (!load_matrix_market(file, A, err)) {

			std::fprintf(stderr, "[spmv_imbal] rank %d failed to load '%s': %s\n", mal_rank(), file, err.c_str());
			MPI_Abort(MPI_COMM_SELF, 1);
		}

	} else {

		A = build_synthetic(pattern, 200000, avg_deg, 12345u, contrast, cfrac);
	}

	const long M = A.rows;
	const long Ncol = A.cols;

	std::vector<float> x((size_t)Ncol);

	for (long c = 0; c < Ncol; c++) {

		x[(size_t)c] = x_value(c);
	}

	const long* row_ptr = A.row_ptr.data();
	const int* col_idx = A.col_idx.data();
	const float* vals = A.vals.data();

	float* y = nullptr;

	if (mal_rank() == 0) {

		y = static_cast<float*>(std::calloc((size_t)M, sizeof(float)));
	}

	if (mal_rank() == 0) {

		#if !BENCH_CSV
		MAL_LOG(MAL_LOG_INFO, "[SETUP] spmv_imbal rows=%ld cols=%ld nnz=%ld max_row_nnz=%ld mean_row_nnz=%.1f reps=%ld src=%s", M, Ncol, A.nnz, A.max_row_nnz(), A.mean_row_nnz(), reps, (synth_n > 0 || file[0] == '\0') ? "synthetic" : file);
		#endif
	}

	const double t0 = MPI_Wtime();
	long row, lim;
	MalFor f = mal_for(M, &row, &lim);

	mal_attach_vec(f, (void**)&y, sizeof(float), M, 0);

	for (; row < lim; row++) {

		const long b = row_ptr[row];
		const long e = row_ptr[row + 1];
		double acc = 0.0;

		for (long rep = 0; rep < reps; rep++) {

			for (long j = b; j < e; j++) {

				acc += (double)vals[j] * (double)x[(size_t)col_idx[j]];
			}
		}

		y[row] = (float)(acc / (double)reps);

		mal_check_for(f);
	}

	mal_finalize();
	const double compute_seconds = MPI_Wtime() - t0;

	#if !BENCH_CSV
	(void)compute_seconds;
	#endif

	if (mal_rank() == 0) {

		#if BENCH_CSV

		print_bench_csv("spmv_imbal", "malleable", "spmv", mal_size(), mal_active_size(), M, compute_seconds, 0);

		#else

		int errors = 0;
		float max_rel = 0.0f;

		for (long r = 0; r < M; r++) {

			double expected = 0.0;

			for (long j = A.row_ptr[(size_t)r]; j < A.row_ptr[(size_t)r + 1]; j++) {

				expected += (double)A.vals[(size_t)j] * (double)x_value(A.col_idx[(size_t)j]);
			}

			const double denom = std::max(1.0, std::fabs(expected));
			const double rel = std::fabs((double)y[r] - expected) / denom;
			max_rel = std::max(max_rel, (float)rel);

			if (rel > 1e-4) {
				errors++;
				if (errors <= 10) {
					MAL_LOG(MAL_LOG_INFO, "[DEBUG] y[%ld]=%.3f expected=%.3f", r, y[r], expected);
				}
			}
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] spmv_imbal %s (rows=%ld nnz=%ld active=%d max_rel=%.2e errors=%d)", errors == 0 ? "OK" : "WRONG", M, A.nnz, mal_active_size(), max_rel, errors);

		#endif
	}

	if (y) {

		std::free(y);
	}

	return EXIT_SUCCESS;
}
