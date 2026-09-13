#include <atomic>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <mpi.h>
#include "malleable.hpp"
#include "example_utils.hpp"

static const char* g_scenario = "";
static int g_abstain = 0;
static std::atomic<int> g_voting_epochs{0};
static std::atomic<int> g_commits_seen{0};

static ResizeDecision scenario_vote(const EpochMetrics& m) {

	ResizeDecision d;
	const int r = mal_rank();
	const int n = m.active_n;

	if (r == 0) {

		g_commits_seen.store(m.resize_commit_count, std::memory_order_relaxed);

	}

	if (m.resize_commit_count > 0) {

		return d;

	}

	if (r == 0) {

		g_voting_epochs.fetch_add(1, std::memory_order_relaxed);

	}

	if (std::strcmp(g_scenario, "unanimous") == 0) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = n + 2;

	} else if (std::strcmp(g_scenario, "outlier") == 0) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = n + (r == 0 ? 1 : r == 1 ? 4 : 2);

	} else if (std::strcmp(g_scenario, "split") == 0) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = (r == n - 1) ? n - 2 : n + 2;

	} else if (std::strcmp(g_scenario, "turnout") == 0) {

		if (r < g_abstain) {

			d.vote = MAL_VOTE_ABSTAIN;

		} else {

			d.vote = MAL_VOTE_RESIZE;
			d.target_active_size = n - 2;

		}

	} else if (std::strcmp(g_scenario, "rebalance") == 0) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = n;

	} else if (std::strcmp(g_scenario, "tie_grow") == 0) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = (r < n / 2) ? n + 1 : n + 3;

	} else if (std::strcmp(g_scenario, "tie_shrink") == 0) {

		d.vote = MAL_VOTE_RESIZE;
		d.target_active_size = (r < n / 2) ? n - 6 : n - 2;

	}

	return d;

}

int main(int argc, char* argv[]) {

	g_scenario = (argc > 1 && argv[1][0] != '-') ? argv[1] : "";
	g_abstain = (int)parse_arg_long(argc, argv, "abstain", 0);
	const long expect_size = parse_arg_long(argc, argv, "expect", -1);
	const long expect_commits = parse_arg_long(argc, argv, "commits", -1);
	const long total = parse_arg_long(argc, argv, "n", 4800);

	mal_set_decide_resize_func(scenario_vote);
	mal_init(MAL_RESIZE_POLICY_CUSTOM);

	long i, limit;
	MalFor f = mal_for(total, i, limit);

	for (; i < limit; i++) {

		usleep(500);
		mal_check_for(f);

	}

	mal_finalize();

	if (mal_rank() == 0) {

		const int final_size = mal_active_size();
		const int commits = g_commits_seen.load(std::memory_order_relaxed);
		const int voting_epochs = g_voting_epochs.load(std::memory_order_relaxed);
		const bool ok = final_size == expect_size && commits == expect_commits && voting_epochs >= (expect_commits == 0 ? 3 : 1);

		MAL_LOG(ok ? MAL_LOG_INFO : MAL_LOG_ERROR, "[RESULT] vote %s %s (size=%d expected=%ld, commits=%d expected=%ld, voting_epochs=%d)", g_scenario, ok ? "OK" : "WRONG", final_size, expect_size, commits, expect_commits, voting_epochs);

	}

	return EXIT_SUCCESS;

}
