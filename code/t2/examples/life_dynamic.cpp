#include "malleable.hpp"
#include "example_utils.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace {

inline uint8_t gol_alive_next(uint8_t self, int neigh) {

	if (self) {

		return (neigh == 2 || neigh == 3) ? 1 : 0;
	}

	return (neigh == 3) ? 1 : 0;
}

void gol_step(const uint8_t* in, uint8_t* out, int G) {

	for (int r = 0; r < G; r++) {

		const int rm = (r == 0) ? G - 1 : r - 1;
		const int rp = (r == G - 1) ? 0 : r + 1;

		for (int c = 0; c < G; c++) {

			const int cm = (c == 0) ? G - 1 : c - 1;
			const int cp = (c == G - 1) ? 0 : c + 1;
			const int n = in[rm * G + cm] + in[rm * G + c] + in[rm * G + cp] + in[r * G + cm] + in[r * G + cp] + in[rp * G + cm] + in[rp * G + c] + in[rp * G + cp];
			out[r * G + c] = gol_alive_next(in[r * G + c], n);
		}
	}
}

long popcount_grid(const uint8_t* g, long n) {

	long s = 0;

	for (long k = 0; k < n; k++) {

		s += g[k];
	}

	return s;
}

void init_grid_random(uint8_t* g, int G, unsigned int seed) {

	unsigned int s = seed;

	for (int k = 0; k < G * G; k++) {

		g[k] = (rand_r(&s) % 100 < 30) ? 1 : 0;
	}
}

}

int main(int argc, char* argv[]) {

	const double period_sec = (double)parse_arg_long(argc, argv, "period_sec", 6);
	const double amp_pct = (double)parse_arg_long(argc, argv, "amp_pct", 100);
	const long jitter_pct = parse_arg_long(argc, argv, "jitter_pct", 3);

	setenv("MAL_FAST_RESPONSE", "1", 0);

	mal_init();

	const long N = parse_arg_long(argc, argv, "n", 50000);
	const long G = parse_arg_long(argc, argv, "grid", 128);
	const long fast_steps = parse_arg_long(argc, argv, "fast_steps", 1);
	const long slow_steps = parse_arg_long(argc, argv, "slow_steps", 12);

	float* data = nullptr;

	if (mal_rank() == 0) {

		data = static_cast<float*>(std::calloc(static_cast<size_t>(N), sizeof(float)));
	}

	std::vector<uint8_t> grid_a(static_cast<size_t>(G * G), 0);
	std::vector<uint8_t> grid_b(static_cast<size_t>(G * G), 0);
	init_grid_random(grid_a.data(), (int)G, (unsigned int)(mal_rank() + 1) * 2654435761u);

	long i, lim;
	MalFor f = mal_for(N, i, lim);

	mal_attach_vec(f, (void**)&data, sizeof(float), N, 0);

	int prev_size = mal_active_size();
	int last_logged_slow_thresh = -1;
	const int universe = mal_size();
	const double t_start = MPI_Wtime();

	unsigned int jitter_state = (unsigned int)(mal_rank() * 2246822519u + 1u);
	auto jittered = [&](long base) -> long {
		if (jitter_pct <= 0 || base <= 0) {

			return base;
		}

		const double r = (double)rand_r(&jitter_state) / (double)RAND_MAX * 2.0 - 1.0;
		const double fac = 1.0 + r * (double)jitter_pct / 100.0;
		double v = (double)base * fac;

		if (v < 0.0) {

			v = 0.0;
		}

		return (long)v;
	};

	if (mal_rank() == 0) {

		MAL_LOG(MAL_LOG_INFO, "[EXPECTED] mode=life_dynamic period=%.1fs universe=%d grid=%ldx%ld", period_sec, universe, G, G);
	}

	for (; i < lim; i++) {

		const double elapsed = MPI_Wtime() - t_start;
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

		const long steps_target = (mal_rank() >= slow_thresh) ? slow_steps : fast_steps;
		const long steps = std::max(0L, jittered(steps_target));

		for (long s = 0; s < steps; s++) {

			gol_step(grid_a.data(), grid_b.data(), (int)G);
			std::swap(grid_a, grid_b);
		}

		const long alive = popcount_grid(grid_a.data(), G * G);
		data[i] = static_cast<float>(alive);

		const int cur_active = mal_active_size();

		if (cur_active != prev_size) {

			MAL_LOG(MAL_LOG_INFO, "[DEMO] *** Resize: %d -> %d ranks (t=%.2fs) ***", prev_size, cur_active, elapsed);
			prev_size = cur_active;
		}

		MAL_LOG(MAL_LOG_INFO, "[DEMO] rank=%d iter=%ld/%ld alive=%ld steps=%ld (active=%d t=%.2fs)", mal_rank(), i, N, alive, steps, cur_active, elapsed);

		mal_check_for(f);
	}

	mal_finalize();

	if (mal_rank() == 0) {

		long total_alive = 0;

		for (long j = 0; j < N; j++) {

			total_alive += (long)data[j];
		}

		MAL_LOG(MAL_LOG_INFO, "[RESULT] life_dynamic OK (sum_alive=%ld N=%ld)", total_alive, N);
		std::free(data);
	}

	return EXIT_SUCCESS;
}
