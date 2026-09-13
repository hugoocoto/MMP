#include "malleable.hpp"

#include <algorithm>
#include <cstring>

namespace builtin_auto {

namespace {

constexpr double kEpsDone = 1.0;

enum class Phase { IDLE, NEEDS_BASELINE, EXPLORE_MAX, SEARCHING, PROBING };
enum class GateAction { Proceed, Defer, Rebalance };

constexpr int kMinBaselineEpochs = 3;
constexpr double kImbHi = 1.50;
constexpr int kImbStreakNeeded = 3;
constexpr int kMaxGateFires = 3;

struct AutoState {

	Phase phase{Phase::IDLE};
	int bs_lo{1};
	int bs_hi{-1};
	int baseline_count{0};
	double thr_single_proc{0.0};
	int gate_fire_streak{0};
	int gate_giveup_at_n{-1};

};

AutoState g_state;
double g_threshold = 0.6;
bool g_baseline_from_perrank = false;

void update_baseline(const EpochMetrics& m) {

	if (m.global_thr <= kEpsThroughput) return;

	if (m.active_n != 1) {

		if (g_baseline_from_perrank && m.active_n > 1 && g_state.thr_single_proc <= kEpsThroughput) {

			g_state.thr_single_proc = m.global_thr / (double)m.active_n;
			g_state.baseline_count = kMinBaselineEpochs;

		}

		return;

	}

	const double min_valid_s = std::max(0.02, m.epoch_interval_ms * 0.0005);

	if (m.epoch_elapsed < min_valid_s) return;

	const bool building = (g_state.phase == Phase::IDLE || g_state.phase == Phase::NEEDS_BASELINE);
	const double alpha = building ? 0.5 : 0.9;

	g_state.thr_single_proc = (g_state.thr_single_proc <= kEpsThroughput)
		? m.global_thr
		: alpha * g_state.thr_single_proc + (1.0 - alpha) * m.global_thr;

	if (building) g_state.baseline_count++;

}

GateAction imbalance_gate(const EpochMetrics& m, bool gate_live, bool in_rebalance_cooldown) {

	const bool imbalanced = (m.active_n > 1) && (m.max_slow_streak >= kImbStreakNeeded);

	if (!gate_live || !mal_get_load_balancing_enabled() || !imbalanced) {

		g_state.gate_fire_streak = 0;
		g_state.gate_giveup_at_n = -1;
		return GateAction::Proceed;

	}

	if (g_state.gate_giveup_at_n == m.active_n) return GateAction::Proceed;
	if (in_rebalance_cooldown) return GateAction::Defer;

	if (g_state.gate_fire_streak >= kMaxGateFires) {

		g_state.gate_giveup_at_n = m.active_n;
		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "imbalance gate: irreducible at N=%d (%d fires, slow_streak=%d) -> allow sizing", m.active_n, g_state.gate_fire_streak, m.max_slow_streak);
		return GateAction::Proceed;

	}

	g_state.gate_fire_streak++;
	return GateAction::Rebalance;

}

ResizeDecision decide_core(const EpochMetrics& m, double threshold) {

	ResizeDecision out;

	if (m.global_remaining < kEpsDone || m.active_n <= 0) {

		out.done = true;
		return out;

	}

	if (m.resize_cooldown_remaining > 0) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Resize skipped: resize_cooldown=%d", m.resize_cooldown_remaining);
		return out;

	}

	const bool in_rebalance_cooldown = (m.rebalance_cooldown_remaining > 0);
	const int U = mal_size();

	if (U == 1) return out;

	update_baseline(m);

	auto compute_efficiency = [&]() -> double {

		if (g_state.thr_single_proc <= kEpsThroughput || m.global_thr <= kEpsThroughput || m.active_n <= 0) return -1.0;

		const double speedup = m.global_thr / g_state.thr_single_proc;
		return speedup / (double)m.active_n;

	};

	{

		const bool gate_live = !mal_get_resize_enabled() || (g_state.phase == Phase::EXPLORE_MAX && m.active_n == U) || g_state.phase == Phase::PROBING;

		switch (imbalance_gate(m, gate_live, in_rebalance_cooldown)) {

			case GateAction::Rebalance:
				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "imbalance gate: ratio=%.2f (>%.2f) -> same-size rebalance N=%d", m.imbalance_ratio(), kImbHi, m.active_n);
				out.should_resize = true;
				out.target_active_size = m.active_n;
				return out;

			case GateAction::Defer:
				return out;

			case GateAction::Proceed:
				break;

		}

	}

	auto log_probe = [&](const char* tag, int probe_n, double eff) {

		if (mal_rank() == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs-%s probe=%d active=%d thr=%.1f thr_1=%.1f speedup=%.2f eff=%.3f threshold=%.2f [lo=%d hi=%d]", tag, probe_n, m.active_n, m.global_thr, g_state.thr_single_proc, (g_state.thr_single_proc > kEpsThroughput ? m.global_thr / g_state.thr_single_proc : 0.0), eff, threshold, g_state.bs_lo, g_state.bs_hi);
			MAL_TRACE_PROBE(0, probe_n, m.active_n, m.global_thr, g_state.thr_single_proc, (g_state.thr_single_proc > kEpsThroughput ? m.global_thr / g_state.thr_single_proc : 0.0), eff);

		}

	};

	if (!mal_get_resize_enabled()) return out;

	switch (g_state.phase) {

	case Phase::IDLE:

		if (g_state.thr_single_proc <= kEpsThroughput) {

			g_state.phase = Phase::NEEDS_BASELINE;
			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: no baseline, going to N=1");

			if (m.active_n != 1) {

				out.should_resize = true;
				out.target_active_size = 1;

			}

			return out;

		}

		g_state.bs_lo = 1;
		g_state.bs_hi = U;
		g_state.phase = Phase::EXPLORE_MAX;

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: baseline thr_1=%.1f, exploring N=%d", g_state.thr_single_proc, U);

		if (m.active_n != U) {

			out.should_resize = true;
			out.target_active_size = U;

		}

		return out;

	case Phase::NEEDS_BASELINE:

		if (g_state.baseline_count < kMinBaselineEpochs || g_state.thr_single_proc <= kEpsThroughput) {

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: baseline %d/%d epochs thr_1=%.1f (waiting)", g_state.baseline_count, kMinBaselineEpochs, g_state.thr_single_proc);
			return out;

		}

		g_state.bs_lo = 1;
		g_state.bs_hi = U;
		g_state.baseline_count = 0;
		g_state.phase = Phase::EXPLORE_MAX;

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: baseline ready thr_1=%.1f (%d epochs), going to N=%d", g_state.thr_single_proc, kMinBaselineEpochs, U);

		out.should_resize = true;
		out.target_active_size = U;

		return out;

	case Phase::EXPLORE_MAX:

		if (m.active_n != U) {

			out.should_resize = true;
			out.target_active_size = U;

			return out;

		}

		{

			const double eff = compute_efficiency();

			if (eff < 0.0) return out;

			log_probe("max", U, eff);

			if (eff >= threshold) {

				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: N=%d meets threshold (eff=%.3f >= %.2f), entering PROBING", U, eff, threshold);
				g_state.phase = Phase::PROBING;

				return out;

			}

			g_state.bs_lo = 1;
			g_state.bs_hi = U;
			const int probe = (g_state.bs_lo + g_state.bs_hi) / 2;
			g_state.phase = Phase::SEARCHING;
			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: N=%d below threshold (eff=%.3f < %.2f), searching [%d,%d] probe=%d", U, eff, threshold, g_state.bs_lo, g_state.bs_hi, probe);

			out.should_resize = true;
			out.target_active_size = probe;

			return out;

		}

	case Phase::SEARCHING:
		{

			const double eff = compute_efficiency();

			if (eff < 0.0) return out;

			log_probe("search", m.active_n, eff);

			if (eff >= threshold) {

				g_state.bs_lo = m.active_n;

			} else {

				g_state.bs_hi = m.active_n;

			}

			if (g_state.bs_hi - g_state.bs_lo <= 1) {

				const int best = g_state.bs_lo;
				g_state.phase = Phase::PROBING;

				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: converged, PROBING from N=%d", best);

				if (best != m.active_n) {

					out.should_resize = true;
					out.target_active_size = best;

				}

				return out;

			}

			const int probe = (g_state.bs_lo + g_state.bs_hi) / 2;

			MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "bs: [%d,%d] next probe=%d", g_state.bs_lo, g_state.bs_hi, probe);

			out.should_resize = true;
			out.target_active_size = probe;

			return out;

		}

	case Phase::PROBING:
		{

			const double eff = compute_efficiency();

			if (eff < 0.0) return out;

			log_probe("probe", m.active_n, eff);

			if (eff >= threshold) {

				if (m.active_n > g_state.bs_lo) {

					g_state.bs_lo = m.active_n;
					MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "probe: improved to bs_lo=%d", g_state.bs_lo);

				}

				if (m.active_n < U) {

					out.should_resize = true;
					out.target_active_size = m.active_n + 1;

					return out;

				}

				return out;

			}

			if (m.active_n > g_state.bs_lo) {

				g_state.bs_hi = m.active_n;
				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "probe: N=%d failed (eff=%.3f < %.2f), returning to bs_lo=%d", m.active_n, eff, threshold, g_state.bs_lo);
				out.should_resize = true;
				out.target_active_size = g_state.bs_lo;

			} else {

				g_state.bs_hi = m.active_n;
				g_state.bs_lo = std::max(1, m.active_n / 2);
				g_state.phase = Phase::SEARCHING;
				const int probe = (g_state.bs_lo + g_state.bs_hi) / 2;

				MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "probe: home N=%d degraded (eff=%.3f < %.2f), re-searching [%d,%d] probe=%d", m.active_n, eff, threshold, g_state.bs_lo, g_state.bs_hi, probe);

				if (probe != m.active_n) {

					out.should_resize = true;
					out.target_active_size = probe;

				}

			}

			return out;

		}

	}

	return out;

}

}

ResizeDecision decide(const EpochMetrics& m) { return decide_core(m, g_threshold); }

ResizeStateBlob save_state() {

	return { &g_state, sizeof(g_state) };

}

void load_state(const void* data, size_t len) {

	if (len != sizeof(g_state)) {

		MAL_LOG_L(MAL_LOG_ERROR, "AUTO", "load_state: got %zu bytes, expected %zu; keeping local state", len, sizeof(g_state));
		return;

	}

	std::memcpy(&g_state, data, len);

}

void install(double threshold) {

	g_threshold = threshold;
	g_baseline_from_perrank = mal_env_bool("MAL_BASELINE_FROM_PERRANK", false);
	mal_set_decide_resize_func(&decide);
	mal_set_decide_resize_state_funcs(&save_state, &load_state);

}

} 
