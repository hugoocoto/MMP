#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {

	const char* so_path = std::getenv("POLICY_SO");
	const char* fn_name = std::getenv("POLICY_FN");

	if (!so_path) {
		so_path = "build/montecarlo_plugin_policy.so";
	}
	if (!fn_name) {
		fn_name = "montecarlo_policy";
	}

	mal_set_decide_resize_plugin(so_path, fn_name);
	mal_init(MAL_RESIZE_POLICY_CUSTOM);

	if (mal_rank() == 0) {
		MAL_LOG(MAL_LOG_INFO, "[PLUGIN] loaded policy \"%s\" from \"%s\"", fn_name, so_path);
	}

	const long total_points = parse_arg_long(argc, argv, "n", 20);
	unsigned int seed = static_cast<unsigned int>(mal_rank() * 2654435761u + 1u);
	const double t0 = MPI_Wtime();

	long i, limit;
	MalFor f = mal_for(total_points, i, limit);

	const useconds_t delay_us = example_delay_us(200000);

	long hits = 0;
	mal_attach_acc(f, hits);

	for (; i < limit; i++) {

		double x = static_cast<double>(rand_r(&seed)) / RAND_MAX;
		double y = static_cast<double>(rand_r(&seed)) / RAND_MAX;

		if (x * x + y * y <= 1.0) {
			hits++;
		}

		MAL_LOG(MAL_LOG_INFO, "[ITER] i=%ld hits_so_far=%ld active=%d",
			i, hits, mal_active_size());
		usleep(delay_us);

		mal_check_for(f);
	}

	mal_finalize();

	if (mal_rank() == 0) {

		const double compute_seconds = MPI_Wtime() - t0;
		const double pi_approx = 4.0 * static_cast<double>(hits) / static_cast<double>(total_points);

		MAL_LOG(MAL_LOG_INFO,
			"[RESULT] montecarlo_plugin OK total_points=%ld hits=%ld "
			"pi~=%.6f error=%.2e t=%.2fs",
			total_points, hits, pi_approx,
			std::fabs(pi_approx - 3.14159265358979),
			compute_seconds);
	}

	return EXIT_SUCCESS;
}
