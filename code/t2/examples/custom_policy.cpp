#include <algorithm>
#include <cstdlib>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

static ResizeDecision my_resize_policy(const EpochMetrics& m) {

	ResizeDecision d;

	const int universe = mal_size();

	if (m.imbalance_ratio() > 1.3 && m.active_n < universe) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = std::min(m.active_n + 2, universe);
		return d;

	}

	if (m.any_settled && m.active_n > 2) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = std::max(m.active_n - 2, 2);
		return d;

	}

	return d;

}

int main(int argc, char* argv[]) {

	const long N = parse_arg_long(argc, argv, "n", 50000);

	mal_set_decide_resize_func(my_resize_policy);
	mal_init(MAL_RESIZE_POLICY_CUSTOM);

	double* data = nullptr;

	if (mal_rank() == 0) {

		data = static_cast<double*>(std::calloc(static_cast<size_t>(N), sizeof(double)));

	}

	long i, lim;
	MalFor f = mal_for(N, i, lim);
	mal_attach_vec(f, (void**)&data, sizeof(double), N, 0);

	for (; i < lim; i++) {

		data[i] = static_cast<double>(i + 1);
		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

		int errors = 0;

		for (long j = 0; j < N; j++) {

			if (data[j] != static_cast<double>(j + 1)) {

				errors++;

			}

		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] custom_policy %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);
		std::free(data);

	}

	return EXIT_SUCCESS;

}
