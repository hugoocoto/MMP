#include "malleable.hpp"

#include <algorithm>
#include <cstring>

namespace builtin_cost {

namespace {

constexpr double kEpsDone = 1.0;

enum class Phase {
	NONE,
	RAMP,
	DESCENT,
	SAMPLE,
	SETTLED
};
enum class GateAction {
	PROCEED,
	DEFER,
	REBALANCE
};

constexpr double kImbHi = 1.50;
constexpr int kImbStreakNeeded = 3;
constexpr int kMaxGateFires = 3;
constexpr int kCostRecheckEpochs = 8;
constexpr int kCostStopStreak = 3;
constexpr int kCostSampleDwell = 1;
constexpr double kCostEwmaAlpha = 0.5;

struct CostState {

	Phase phase{Phase::NONE};
	double best_g{0.0};
	int best_n{0};
	double prev_g{0.0};
	int prev_n{-1};
	int settle_recheck{0};
	int stop_streak{0};

	int sample_target{0};
	int sample_min{0};
	int sample_step{0};
	int sample_coarse_step{0};
	bool sample_fine{false};
	int sample_dwell_left{0};
	int sample_meas_left{0};
	double sample_thr_accum{0.0};
	double sample_best_thr{0.0};
	int sample_best_n{0};

	int gate_fire_streak{0};
	int gate_giveup_at_n{-1};

	bool cooldown_exempt{false};
};

struct CostConfig {

	double keep_fraction{0.97};
	int sample_step{8};
	bool sample_refine{false};
	int sample_meas{3};
};

CostState g_state;
CostConfig g_config;

GateAction imbalance_gate(const EpochMetrics& m, bool gate_live, bool in_rebalance_cooldown) {

	const bool imbalanced = (m.active_n > 1) && (m.max_slow_streak >= kImbStreakNeeded);

	if (!gate_live || !mal_get_load_balancing_enabled() || !imbalanced) {

		g_state.gate_fire_streak = 0;
		g_state.gate_giveup_at_n = -1;
		return GateAction::PROCEED;
	}

	if (g_state.gate_giveup_at_n == m.active_n) {
		return GateAction::PROCEED;
	}
	if (in_rebalance_cooldown) {
		return GateAction::DEFER;
	}

	if (g_state.gate_fire_streak >= kMaxGateFires) {

		g_state.gate_giveup_at_n = m.active_n;
		MAL_LOG_L(MAL_LOG_DEBUG, "COST", "imbalance gate: irreducible at N=%d (%d fires, slow_streak=%d) -> allow sizing", m.active_n, g_state.gate_fire_streak, m.max_slow_streak);
		return GateAction::PROCEED;
	}

	g_state.gate_fire_streak++;
	return GateAction::REBALANCE;
}

}

ResizeStateBlob save_state() {

	return {&g_state, sizeof(g_state)};
}

void load_state(const void* data, size_t len) {

	if (len != sizeof(g_state)) {

		MAL_LOG_L(MAL_LOG_ERROR, "COST", "load_state: got %zu bytes, expected %zu; keeping local state", len, sizeof(g_state));
		return;
	}

	std::memcpy(&g_state, data, len);
}

ResizeDecision decide_phase(const EpochMetrics& m) {

	ResizeDecision out;

	if (m.global_remaining < kEpsDone || m.active_n <= 0) {

		out.done = true;
		return out;
	}

	const int U = mal_size();

	if (U == 1) {
		return out;
	}

	const int N = m.active_n;

	if (m.global_thr <= kEpsThroughput) {
		return out;
	}

	const double imb = m.imbalance_ratio();
	const double thr_inst = (imb > 1.0) ? m.global_thr / imb : m.global_thr;

	if (N != g_state.prev_n) {

		g_state.prev_g = 0.0;
		g_state.prev_n = N;
		g_state.gate_fire_streak = 0;
		g_state.gate_giveup_at_n = -1;
	}

	const double thr = (g_state.prev_g > kEpsThroughput) ? kCostEwmaAlpha * g_state.prev_g + (1.0 - kCostEwmaAlpha) * thr_inst : thr_inst;
	g_state.prev_g = thr;

	if (m.resize_cooldown_remaining > 0 && g_state.phase != Phase::SAMPLE && !g_state.cooldown_exempt) {

		MAL_LOG_L(MAL_LOG_DEBUG, "COST", "Resize skipped: resize_cooldown=%d", m.resize_cooldown_remaining);
		return out;
	}

	const double keep = g_config.keep_fraction;

	if (g_state.phase != Phase::RAMP && g_state.phase != Phase::DESCENT && g_state.phase != Phase::SAMPLE && g_state.phase != Phase::SETTLED) {

		g_state.phase = Phase::RAMP;
		g_state.best_g = 0.0;
		g_state.best_n = U;
		g_state.prev_g = 0.0;
	}

	{

		const bool cost_gate_live = !(g_state.phase == Phase::RAMP && N != U) && g_state.phase != Phase::SAMPLE;
		const bool cost_in_reb_cd = (m.rebalance_cooldown_remaining > 0);

		switch (imbalance_gate(m, cost_gate_live, cost_in_reb_cd)) {

			case GateAction::REBALANCE:
				MAL_LOG_L(MAL_LOG_DEBUG, "COST", "imbalance gate: ratio=%.2f (>%.2f) -> same-size rebalance N=%d", m.imbalance_ratio(), kImbHi, N);
				g_state.prev_g = 0.0;

				if (g_state.phase == Phase::RAMP) {
					g_state.best_g = 0.0;
				}

				out.vote = MAL_VOTE_RESIZE;
				out.target_active_size = N;
				return out;

			case GateAction::DEFER:
				return out;

			case GateAction::PROCEED:
				break;
		}
	}

	auto log_cost = [&](const char* tag) {
		if (mal_rank() == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "COST", "%s N=%d thr=%.1f peak=%.1f floor=%.1f keep=%.2f best_n=%d", tag, N, thr, g_state.best_g, g_state.best_g * keep, keep, g_state.best_n);
		}
	};

	if (!mal_get_resize_enabled()) {
		return out;
	}

	switch (g_state.phase) {

		case Phase::RAMP:

			if (N != U) {

				out.vote = MAL_VOTE_RESIZE;
				out.target_active_size = U;
				return out;
			}

			if (g_state.best_g <= kEpsThroughput) {

				g_state.best_g = thr;
				log_cost("ramp-warmup");
				return out;
			}

			g_state.best_g = thr;
			g_state.best_n = U;
			g_state.stop_streak = 0;

			if (m.iterative_kernel && U > 1) {

				const int coarse_step = g_config.sample_step;
				g_state.sample_best_thr = thr;
				g_state.sample_best_n = U;
				g_state.sample_coarse_step = coarse_step;
				g_state.sample_step = coarse_step;
				g_state.sample_fine = false;
				g_state.sample_min = std::max(1, U / 4);
				g_state.sample_target = std::max(g_state.sample_min, U - coarse_step);
				g_state.sample_dwell_left = kCostSampleDwell;
				g_state.sample_meas_left = g_config.sample_meas;
				g_state.sample_thr_accum = 0.0;
				g_state.phase = Phase::SAMPLE;
				log_cost("sample-start");
				out.vote = MAL_VOTE_RESIZE;
				out.target_active_size = g_state.sample_target;
				return out;
			}

			g_state.phase = Phase::DESCENT;
			log_cost("ramp-peak");

			if (U > 1) {

				out.vote = MAL_VOTE_RESIZE;
				out.target_active_size = U - 1;

			} else {

				g_state.phase = Phase::SETTLED;
				g_state.settle_recheck = kCostRecheckEpochs;
			}

			return out;

		case Phase::DESCENT: {

			if (thr > g_state.best_g) {
				g_state.best_g = thr;
			}

			if (N >= g_state.best_n) {

				if (N > 1) {

					out.vote = MAL_VOTE_RESIZE;
					out.target_active_size = N - 1;

				} else {

					g_state.phase = Phase::SETTLED;
					g_state.settle_recheck = kCostRecheckEpochs;
				}

				return out;
			}

			const double floor = g_state.best_g * keep;

			if (thr >= floor) {

				g_state.best_n = N;
				g_state.stop_streak = 0;
				log_cost("descend-keep");

				if (N > 1) {

					out.vote = MAL_VOTE_RESIZE;
					out.target_active_size = N - 1;

				} else {

					g_state.phase = Phase::SETTLED;
					g_state.settle_recheck = kCostRecheckEpochs;
				}

				return out;
			}

			g_state.stop_streak++;

			if (m.iterative_kernel && g_state.stop_streak < kCostStopStreak && N > 1) {

				log_cost("descend-probe");
				out.vote = MAL_VOTE_RESIZE;
				out.target_active_size = N - 1;

				return out;
			}

			log_cost("descend-stop");
			g_state.stop_streak = 0;
			g_state.phase = Phase::SETTLED;
			g_state.settle_recheck = kCostRecheckEpochs;
			out.vote = MAL_VOTE_RESIZE;
			out.target_active_size = g_state.best_n;

			return out;
		}

		case Phase::SAMPLE: {

			if (N != g_state.sample_target) {

				out.vote = MAL_VOTE_RESIZE;
				out.target_active_size = g_state.sample_target;
				return out;
			}

			if (g_state.sample_dwell_left > 0) {

				g_state.sample_dwell_left--;
				return out;
			}

			g_state.sample_thr_accum += thr_inst;
			g_state.sample_meas_left--;

			if (g_state.sample_meas_left > 0) {
				return out;
			}

			const double cand_thr = g_state.sample_thr_accum / (double)g_config.sample_meas;

			if (mal_rank() == 0) {

				MAL_LOG_L(MAL_LOG_DEBUG, "COST", "sample N=%d thr=%.1f best_n=%d best_thr=%.1f", g_state.sample_target, cand_thr, g_state.sample_best_n, g_state.sample_best_thr);
			}

			if (cand_thr > g_state.sample_best_thr) {

				g_state.sample_best_thr = cand_thr;
				g_state.sample_best_n = g_state.sample_target;
			}

			const int next = g_state.sample_target - std::max(1, g_state.sample_step);

			if (next >= g_state.sample_min) {

				g_state.sample_target = next;
				g_state.sample_dwell_left = kCostSampleDwell;
				g_state.sample_meas_left = g_config.sample_meas;
				g_state.sample_thr_accum = 0.0;

				out.vote = MAL_VOTE_RESIZE;
				out.target_active_size = next;
				return out;
			}

			if (!g_state.sample_fine && g_config.sample_refine) {

				const int cstep = std::max(1, g_state.sample_coarse_step);
				const int fstep = std::max(1, cstep / 4);
				const int fine_lo = std::max(1, g_state.sample_best_n - cstep);
				const int fine_hi = std::min(U, g_state.sample_best_n + cstep);

				if (fstep < cstep && fine_hi - fine_lo >= fstep) {

					g_state.sample_fine = true;
					g_state.sample_step = fstep;
					g_state.sample_min = fine_lo;
					g_state.sample_target = fine_hi;
					g_state.sample_dwell_left = kCostSampleDwell;
					g_state.sample_meas_left = g_config.sample_meas;
					g_state.sample_thr_accum = 0.0;

					if (mal_rank() == 0) {

						MAL_LOG_L(MAL_LOG_DEBUG, "COST", "coarse done (argmax N=%d) -> fine [%d..%d] step=%d", g_state.sample_best_n, fine_lo, fine_hi, fstep);
					}

					out.vote = MAL_VOTE_RESIZE;
					out.target_active_size = fine_hi;
					return out;
				}
			}

			g_state.best_n = g_state.sample_best_n;
			g_state.best_g = g_state.sample_best_thr;
			g_state.phase = Phase::SETTLED;
			g_state.settle_recheck = kCostRecheckEpochs;

			if (mal_rank() == 0) {

				MAL_LOG_L(MAL_LOG_INFO, "COST", "sample done -> N*=%d (thr=%.1f)", g_state.sample_best_n, g_state.sample_best_thr);
			}

			out.vote = MAL_VOTE_RESIZE;
			out.target_active_size = g_state.sample_best_n;
			return out;
		}

		case Phase::SETTLED:

			if (m.iterative_kernel) {
				return out;
			}

			if (g_state.settle_recheck > 0) {

				g_state.settle_recheck--;
				return out;
			}

			log_cost("recheck");
			g_state.phase = Phase::RAMP;
			g_state.best_g = 0.0;
			g_state.prev_g = 0.0;
			return out;

		default:
			return out;
	}
}

ResizeDecision decide(const EpochMetrics& m) {

	ResizeDecision out = decide_phase(m);

	if (out.vote == MAL_VOTE_RESIZE && out.target_active_size != m.active_n) {

		g_state.cooldown_exempt = (g_state.phase == Phase::SAMPLE);
	}

	out.settled = (g_state.phase == Phase::SETTLED);
	return out;
}

void install() {

	g_config.keep_fraction = mal_env_double("MAL_COST_KEEP_FRACTION", 0.97, 0.01, 1.0);
	g_config.sample_step = (int)mal_env_long("MAL_COST_SAMPLE_STEP", 8, 1, INT_MAX);
	g_config.sample_refine = mal_env_bool("MAL_COST_SAMPLE_REFINE", false);
	g_config.sample_meas = (int)mal_env_long("MAL_COST_SAMPLE_MEAS", 3, 1, INT_MAX);
	mal_set_decide_resize_func(&decide);
	mal_set_decide_resize_state_funcs(&save_state, &load_state);
}

}
