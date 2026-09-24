#include "malleable.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace builtin_fixed_sequence {

namespace {

std::vector<int> parse_sequence(const char* text) {

	std::vector<int> seq;

	if (!text || !*text) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "Missing resize sequence: set MAL_RESIZE_SEQ");
		std::abort();
	}

	const char* p = text;

	while (*p) {

		char* end = nullptr;
		long n = std::strtol(p, &end, 10);

		if (end == p || n <= 0) {

			MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "MAL_RESIZE_SEQ is invalid; expected comma-separated positive integers");
			std::abort();
		}

		seq.push_back((int)n);
		p = end;

		while (*p == ',' || *p == ' ') {
			p++;
		}
	}

	if (seq.empty()) {

		MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "MAL_RESIZE_SEQ is empty");
		std::abort();
	}

	for (int target : seq) {

		if (target > mal_size()) {

			MAL_LOG_L(MAL_LOG_ERROR, "CONFIG", "Resize target %d in MAL_RESIZE_SEQ exceeds universe size=%d", target, mal_size());
			std::abort();
		}
	}

	std::vector<int> distinct;

	for (int target : seq) {

		if (distinct.empty() || distinct.back() != target) {
			distinct.push_back(target);
		}
	}

	return distinct;
}

std::vector<int> g_seq;

}

ResizeDecision decide(const EpochMetrics& m) {

	ResizeDecision out;

	size_t i = (size_t)m.resize_commit_count;

	if (i < g_seq.size() && g_seq[i] == m.active_n) {
		i++;
	}

	if (i >= g_seq.size()) {

		out.done = true;
		return out;
	}

	out.vote = MAL_VOTE_RESIZE;
	out.target_active_size = g_seq[i];
	out.skip_cooldown = true;

	return out;
}

void install() {

	g_seq = parse_sequence(std::getenv("MAL_RESIZE_SEQ"));
	mal_set_resize_min_horizon_epochs(0);
	mal_set_decide_resize_func(&decide);
}

}
