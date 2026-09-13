#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char* argv[]) {

	const bool dynamic = parse_arg_long(argc, argv, "dynamic", 0) != 0;
	const double period_sec = (double)parse_arg_long(argc, argv, "period_sec", 8);
	const double amp_pct = (double)parse_arg_long(argc, argv, "amp_pct", 60);

	if (dynamic) {

		setenv("MAL_FAST_RESPONSE", "1", 0);
	}

	mal_init();

	const long N = parse_arg_long(argc, argv, "n", 20000);
	const double phase2_sec = (double)parse_arg_long(argc, argv, "phase2", 2);
	const long fast_us = parse_arg_long(argc, argv, "fast_us", 2000);
	const long slow_us = parse_arg_long(argc, argv, "slow_us", 2000000);
	const int slow_rank = (int)parse_arg_long(argc, argv, "slow_rank", 3);

	const long jitter_pct = parse_arg_long(argc, argv, "jitter_pct", 0);
	unsigned int rng_state = (unsigned int)(mal_rank() * 2654435761u + 1u);

	auto jittered_us = [&](long base_us) -> useconds_t {
		if (jitter_pct <= 0 || base_us <= 0) {

			return (useconds_t)base_us;
		}

		const double r = (double)rand_r(&rng_state) / (double)RAND_MAX * 2.0 - 1.0;
		const double factor = 1.0 + r * (double)jitter_pct / 100.0;
		double v = (double)base_us * factor;

		if (v < 0.0) {

			v = 0.0;
		}

		return (useconds_t)v;
	};

	float* data = nullptr;

	if (mal_rank() == 0) {

		data = static_cast<float*>(std::calloc(static_cast<size_t>(N), sizeof(float)));
	}

	long i, lim;
	MalFor f = mal_for(N, i, lim);

	mal_attach_vec(f, (void**)&data, sizeof(float), N, 0);

	int prev_size = mal_active_size();
	double t_start = MPI_Wtime();
	bool phase_logged_up = false;
	bool phase_logged_down = false;
	int last_logged_slow_thresh = -1;
	const int universe = mal_size();

	if (mal_rank() == 0) {

		if (dynamic) {

			MAL_LOG(MAL_LOG_INFO, "[EXPECTED] mode=dynamic period=%.1fs universe=%d", period_sec, universe);

		} else {

			MAL_LOG(MAL_LOG_INFO, "[EXPECTED] mode=static phase=scale-up target=%d reason=all_ranks_fast", universe);
		}
	}

	for (; i < lim; i++) {

		data[i] = static_cast<float>(i + 1);

		double elapsed = MPI_Wtime() - t_start;

		if (dynamic) {

			constexpr double kTwoPi = 6.28318530717958647692;
			const double phase_rad = kTwoPi * elapsed / period_sec;
			const double amp = std::clamp(amp_pct, 0.0, 100.0) / 100.0;

			const double slow_fraction = 0.5 + 0.5 * amp * std::sin(phase_rad);

			int slow_thresh = (int)std::round((1.0 - slow_fraction) * (double)universe);

			if (slow_thresh < 1) {

				slow_thresh = 1;
			}

			if (slow_thresh > universe) {

				slow_thresh = universe;
			}

			if (mal_rank() == 0 && slow_thresh != last_logged_slow_thresh) {

				MAL_LOG(MAL_LOG_INFO, "[EXPECTED] event=phase_change t_rel=%.4f slow_fraction=%.2f slow_thresh=%d active=%d", elapsed, slow_fraction, slow_thresh, mal_active_size());
				last_logged_slow_thresh = slow_thresh;
			}

			if (mal_rank() >= slow_thresh) {

				usleep(jittered_us(slow_us));

			} else {

				usleep(jittered_us(fast_us));
			}

		} else {

			const bool in_scaleup = (elapsed < phase2_sec);

			if (in_scaleup) {

				if (!phase_logged_up && mal_rank() == 0) {

					MAL_LOG(MAL_LOG_INFO, "[EXPECTED] event=phase_start phase=scale-up t_rel=%.4f target=%d", elapsed, mal_size());
					phase_logged_up = true;
				}

				usleep(jittered_us(fast_us));

			} else {

				if (!phase_logged_down && mal_rank() == 0) {

					MAL_LOG(MAL_LOG_INFO, "[EXPECTED] event=phase_start phase=scale-down t_rel=%.4f target=%d reason=ranks_geq_%d_slow", elapsed, slow_rank, slow_rank);
					phase_logged_down = true;
				}

				if (mal_rank() >= slow_rank) {

					usleep(jittered_us(slow_us));

				} else {

					usleep(jittered_us(fast_us));
				}
			}
		}

		const int cur_active = mal_active_size();

		if (cur_active != prev_size) {

			MAL_LOG(MAL_LOG_INFO, "[DEMO] *** Resize: %d -> %d ranks (t=%.2fs) ***", prev_size, cur_active, elapsed);
			prev_size = cur_active;
		}

		MAL_LOG(MAL_LOG_INFO, "[DEMO] rank=%d iter=%ld/%ld done (active=%d t=%.2fs)", mal_rank(), i, N, cur_active, elapsed);

		mal_check_for(f);
	}

	mal_finalize();

	if (mal_rank() == 0) {

		int errors = 0;

		for (long j = 0; j < N; j++) {

			if (data[j] != static_cast<float>(j + 1)) {

				if (errors < 20) {

					MAL_LOG(MAL_LOG_INFO, "[DEBUG] data[%ld]=%.1f expected=%.1f", j, data[j], (float)(j + 1));
				}

				errors++;
			}
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] resize_demo %s (%d errors)", errors == 0 ? "OK" : "WRONG", errors);
		std::free(data);
	}

	return EXIT_SUCCESS;
}
