#include "malleable.hpp"
#include <algorithm>

extern "C" ResizeDecision montecarlo_policy(const EpochMetrics& m) {

	ResizeDecision d;

	if (m.imbalance_ratio() > 1.3) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = m.active_n + 1;
		return d;
	}

	if (m.any_settled && m.active_n > 2) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = m.active_n - 1;
		return d;
	}

	return d;
}
