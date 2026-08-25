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

		while (*p == ',' || *p == ' ') p++;

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

		if (distinct.empty() || distinct.back() != target) distinct.push_back(target);

	}

	return distinct;

}

const std::vector<int>& sequence() {

	static const bool disabled_horizon_throttle = [] {
		mal_set_resize_min_horizon_epochs(0);
		return true;
	}();
	(void)disabled_horizon_throttle;

	static const std::vector<int> seq = parse_sequence(std::getenv("MAL_RESIZE_SEQ"));
	return seq;

}

}

ResizeDecision decide(const EpochMetrics& m) {

	ResizeDecision out;

	const std::vector<int>& seq = sequence();
	const int i = m.resize_commit_count;

	if (i < 0 || (size_t)i >= seq.size()) {

		out.done = true;
		return out;

	}

	const int target = seq[(size_t)i];

	if (target == m.active_n) return out; 

	out.should_resize = true;
	out.target_active_size = target;

	return out;

}

} 
