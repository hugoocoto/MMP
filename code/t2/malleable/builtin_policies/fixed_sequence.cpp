// fixed_sequence.cpp - the library's built-in "replay a prescribed
// trajectory" resize policy, compiled directly into libmalleable.a and
// registered by mal_init() via mal_set_decide_resize_func() (see
// malleable_runtime.cpp) -- same decide()-plugin contract an
// experiments/decision/*.cpp plugin uses, just linked in rather than
// dlopen'd, since it's a built-in known at compile time. Deliberately
// stateless: position in the sequence is derived fresh every call from
// EpochMetrics.resize_commit_count (identical on every universe rank every
// epoch, including a rank that just became active), rather than
// hand-rolled cross-call state -- see the design plan for why a naive
// f(active_n)-only reformulation is ambiguous for sequences with
// non-consecutive repeats (e.g. "4,8,4,2"), and how resize_commit_count
// resolves that.
//
// One deliberate behavior change from the old in-runtime implementation:
// on a non-unanimous epoch, this retries the same target next epoch
// instead of giving up and skipping ahead to the next sequence entry --
// the intended step still eventually happens rather than being silently
// abandoned.

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

	// Collapse only *consecutive* duplicates -- a non-consecutive repeat
	// (e.g. "4,8,4,2") is a legitimate prescribed trajectory (oscillation
	// benchmarks) and must stay distinct, since it's exactly the case that
	// makes a plain f(active_n) lookup ambiguous.
	std::vector<int> distinct;

	for (int target : seq) {

		if (distinct.empty() || distinct.back() != target) distinct.push_back(target);

	}

	return distinct;

}

const std::vector<int>& sequence() {

	// A prescribed trajectory must keep advancing even very close to the
	// end of a run, or a short benchmark would never visit its tail --
	// disable the runtime's generic "don't resize with <2 epochs of
	// estimated runtime left" throttle for this policy specifically.
	static const bool disabled_horizon_throttle = [] {
		mal_set_resize_min_horizon_epochs(0);
		return true;
	}();
	(void)disabled_horizon_throttle;

	static const std::vector<int> seq = parse_sequence(std::getenv("MAL_RESIZE_SEQ"));
	return seq;

}

} // namespace

ResizeDecision decide(const EpochMetrics& m) {

	ResizeDecision out;

	const std::vector<int>& seq = sequence();
	const int i = m.resize_commit_count;

	if (i < 0 || (size_t)i >= seq.size()) {

		out.done = true;
		return out;

	}

	const int target = seq[(size_t)i];

	if (target == m.active_n) return out; // already there, nothing to do

	out.should_resize = true;
	out.target_active_size = target;

	return out;

}

} // namespace builtin_fixed_sequence
