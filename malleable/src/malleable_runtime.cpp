#include "malleable_internal.hpp"

#include <csignal>
#include <cstdlib>
#include <string>

static void log_mpi_error(const char* where, int rc) {

	if (rc == MPI_SUCCESS) {

		return;
	}

	char err[MPI_MAX_ERROR_STRING] = {};
	int len = 0;
	MPI_Error_string(rc, err, &len);
	MAL_LOG_L(MAL_LOG_ERROR, "MPI", "%s failed rc=%d msg=%.*s", where, rc, len, err);
}

static bool parse_env_bool(const char* text, bool& out) {

	char* end = nullptr;
	const long n = std::strtol(text, &end, 10);

	if (end != text && *end == '\0') {

		out = (n != 0);
		return true;
	}

	if (strcasecmp(text, "true") == 0 || strcasecmp(text, "on") == 0 || strcasecmp(text, "yes") == 0) {

		out = true;
		return true;
	}

	if (strcasecmp(text, "false") == 0 || strcasecmp(text, "off") == 0 || strcasecmp(text, "no") == 0) {

		out = false;
		return true;
	}

	return false;
}

bool mal_env_bool(const char* name, bool fallback) {

	const char* v = std::getenv(name);
	bool val = fallback;

	if (!v) {

		return fallback;
	}

	if (!parse_env_bool(v, val)) {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring %s='%s' (valid bool), using %d", name, v, (int)fallback);
		return fallback;
	}

	MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "%s=%d", name, (int)val);
	return val;
}

long mal_env_long(const char* name, long fallback, long min, long max) {

	const char* v = std::getenv(name);

	if (!v) {

		return fallback;
	}

	char* end = nullptr;
	const long val = std::strtol(v, &end, 10);

	if (end == v || *end != '\0' || val < min || val > max) {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring %s='%s' (must be an integer in [%ld, %ld]), using %ld", name, v, min, max, fallback);
		return fallback;
	}

	MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "%s=%ld", name, val);
	return val;
}

double mal_env_double(const char* name, double fallback, double min, double max) {

	const char* v = std::getenv(name);

	if (!v) {

		return fallback;
	}

	char* end = nullptr;
	const double val = std::strtod(v, &end);

	if (end == v || *end != '\0' || !(val >= min && val <= max)) {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring %s='%s' (must be a number in [%g, %g]), using %g", name, v, min, max, fallback);
		return fallback;
	}

	MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "%s=%g", name, val);
	return val;
}

static bool parse_env_log_level(const char* text, MalLogLevel& out) {

	char* end = nullptr;
	const long n = std::strtol(text, &end, 10);

	if (end != text && *end == '\0') {

		if (n < MAL_LOG_DEBUG || n > MAL_LOG_NONE) {

			return false;
		}

		out = (MalLogLevel)n;
		return true;
	}

	for (int level = MAL_LOG_DEBUG; level <= MAL_LOG_NONE; level++) {

		if (strcasecmp(text, mal_log_level_name((MalLogLevel)level)) == 0) {

			out = (MalLogLevel)level;
			return true;
		}
	}

	if (strcasecmp(text, "warning") == 0) {

		out = MAL_LOG_WARN;
		return true;
	}

	return false;
}

void mal_set_epoch_interval_ms(int ms) {

	if (ms > 0) {

		g.cfg.epoch_ms.store(ms);

	} else {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring invalid epoch interval ms=%d (must be > 0)", ms);
	}

	g.sync.notify();
}

void mal_set_resize_enabled(bool b) {

	g.cfg.enabled.store(b);

	if (!b) {

		clear_prepared_resize();
	}

	g.sync.notify();
}

bool mal_get_resize_enabled() {

	return g.cfg.enabled.load(std::memory_order_relaxed);
}

bool mal_get_load_balancing_enabled() {

	return g.cfg.load_balancing_enabled.load(std::memory_order_relaxed);
}

void mal_set_resize_min_horizon_epochs(int epochs) {

	g.cfg.resize_min_horizon_epochs.store((double)epochs, std::memory_order_relaxed);
}

void mal_set_resize_quorum(double quorum) {

	if (!(quorum > 0.5 && quorum <= 1.0)) {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring resize quorum=%g (must be in (0.5, 1])", quorum);
		return;
	}

	g.cfg.resize_quorum.store(quorum, std::memory_order_relaxed);
}

void mal_set_attach_exec_mode(MalAttachExecMode mode) {

	if (mode != MAL_ATTACH_SYNC && mode != MAL_ATTACH_ASYNC) {

		MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring invalid attach execution mode=%d", (int)mode);

		return;
	}

	g.cfg.attach_mode.store(mode);
	g.sync.notify();
}

MalAttachExecMode mal_get_attach_exec_mode() {

	return g.cfg.attach_mode.load();
}

void load_env_config() {

	if (const char* v = std::getenv("MAL_LOG_LEVEL")) {

		MalLogLevel level;

		if (parse_env_log_level(v, level)) {

			g.cfg.log_level.store(level);
			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_LOG_LEVEL=%s", mal_log_level_name(level));

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_LOG_LEVEL='%s' (valid: DEBUG/INFO/WARN/ERROR/NONE or 0..4)", v);
		}
	}

	g.cfg.trace_enabled = mal_env_bool("MAL_TRACE_ENABLED", g.cfg.trace_enabled);
	g.cfg.log_all_ranks.store(mal_env_bool("MAL_LOG_ALL_RANKS", g.cfg.log_all_ranks.load()));

	if (const char* v = std::getenv("MAL_RESIZE_POLICY")) {

		static constexpr struct {
			const char* name;
			MalResizePolicy policy;
		} kPolicies[] = {
			{"auto", MAL_RESIZE_POLICY_AUTO},
			{"throughput", MAL_RESIZE_POLICY_THROUGHPUT},
			{"efficiency", MAL_RESIZE_POLICY_EFFICIENCY},
			{"energy", MAL_RESIZE_POLICY_EFFICIENCY},
			{"fixed", MAL_RESIZE_POLICY_FIXED_SEQUENCE},
			{"fixed_sequence", MAL_RESIZE_POLICY_FIXED_SEQUENCE},
			{"cost", MAL_RESIZE_POLICY_COST},
		};

		bool ok = false;

		for (const auto& p : kPolicies) {

			if (strcasecmp(v, p.name) == 0) {

				g.cfg.resize_policy = p.policy;
				ok = true;
				break;
			}
		}

		if (ok) {

			MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "MAL_RESIZE_POLICY=%s", v);

		} else if (strcasecmp(v, "balance") == 0 || strcasecmp(v, "lb") == 0) {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "MAL_RESIZE_POLICY=balance/lb was REMOVED — use MAL_RESIZE_ENABLED=0 MAL_LOAD_BALANCING_ENABLED=1 (any policy) for fixed-N rebalancing. Ignoring.");

		} else {

			MAL_LOG_L(MAL_LOG_WARN, "CONFIG", "Ignoring MAL_RESIZE_POLICY='%s' (valid: auto/throughput/energy/fixed/cost)", v);
		}
	}

	g.cfg.epoch_ms.store((int)mal_env_long("MAL_EPOCH_INTERVAL_MS", g.cfg.epoch_ms.load(), 1, INT_MAX));
	g.cfg.stencil_resid_reduces.store((int)mal_env_long("MAL_STENCIL_RESID_REDUCES", g.cfg.stencil_resid_reduces.load(), 1, INT_MAX));
	g.cfg.stencil_epoch_steps.store((int)mal_env_long("MAL_STENCIL_EPOCH_STEPS", g.cfg.stencil_epoch_steps.load(), 1, INT_MAX));
	g.cfg.epoch_change_mode.store((int)mal_env_long("MAL_EPOCH_CHANGE_MODE", g.cfg.epoch_change_mode.load(), MAL_EPOCH_CHANGE_RECALCULATE, MAL_EPOCH_CHANGE_USE_LAST_DECISION));
	mal_set_resize_quorum(mal_env_double("MAL_RESIZE_QUORUM", g.cfg.resize_quorum.load(), 0.0, 1.0));

	g.cfg.enabled.store(mal_env_bool("MAL_RESIZE_ENABLED", g.cfg.enabled.load()));
	g.cfg.malleability_enabled.store(mal_env_bool("MAL_MALLEABILITY_ENABLED", g.cfg.malleability_enabled.load()));
	g.cfg.load_balancing_enabled.store(mal_env_bool("MAL_LOAD_BALANCING_ENABLED", g.cfg.load_balancing_enabled.load()));

	if (!g.cfg.malleability_enabled.load()) {

		g.cfg.enabled.store(false);
		MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "Malleability disabled: forcing MAL_RESIZE_ENABLED=0");
	}

	g.cfg.fast_response.store(mal_env_bool("MAL_FAST_RESPONSE", g.cfg.fast_response.load()));
	g.cfg.affinity_enabled = mal_env_bool("MAL_AFFINITY", g.cfg.affinity_enabled);
	g.cfg.main_core = (int)mal_env_long("MAL_MAIN_CORE", g.cfg.main_core, 0, INT_MAX);
	g.cfg.worker_core = (int)mal_env_long("MAL_WORKER_CORE", g.cfg.worker_core, 0, INT_MAX);
	g.cfg.initial_size = (int)mal_env_long("MAL_INITIAL_SIZE", g.cfg.initial_size, 1, INT_MAX);
	g.timing.enabled = mal_env_bool("MAL_TIMING", g.timing.enabled);
}

inline bool active_comm_ready_or_stop() {

	return g.comm.active != MPI_COMM_NULL || g.sync.stop.load(std::memory_order_acquire);
}

inline bool attach_pending_cleared_or_stop() {

	return !g.sync.attach_pending.load(std::memory_order_acquire) || g.sync.stop.load(std::memory_order_acquire);
}

inline bool resize_pending_cleared_or_stop() {

	return !g.sync.resize_pending.load(std::memory_order_acquire) || g.sync.stop.load(std::memory_order_acquire);
}

void mal_init(MalResizePolicy policy) {

	if (policy == MAL_RESIZE_POLICY_CUSTOM && g.cfg.decide_resize_func == nullptr) {

		policy = MAL_RESIZE_POLICY_AUTO;
	}

	g.cfg.resize_policy = policy;

	signal(SIGPIPE, SIG_IGN);

	load_env_config();

	if (g.cfg.resize_policy == MAL_RESIZE_POLICY_CUSTOM) {

		MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "decide function pre-registered, resize_policy=CUSTOM");
	}

	setenv("OMPI_MCA_coll_han_priority", "0", 0);
	setenv("OMPI_MCA_coll_adapt_priority", "0", 0);

	MPI_Info init_info = MPI_INFO_NULL;
	MPI_Info_create(&init_info);
	MPI_Info_set(init_info, "thread_level", "MPI_THREAD_MULTIPLE");

	int rc = MPI_Session_init(init_info, MPI_ERRORS_RETURN, &g.comm.session);

	MPI_Info_free(&init_info);

	if (rc != MPI_SUCCESS) {

		log_mpi_error("MPI_Session_init", rc);
		std::abort();
	}

	const double t_init_start = MPI_Wtime();
	g.timing.t_origin = t_init_start;

	rc = MPI_Group_from_session_pset(g.comm.session, "mpi://WORLD", &g.comm.world_group);

	if (rc != MPI_SUCCESS) {

		log_mpi_error("MPI_Group_from_session_pset", rc);
		std::abort();
	}

	rc = MPI_Comm_create_from_group(g.comm.world_group, "malleable.universe", MPI_INFO_NULL, MPI_ERRORS_RETURN, &g.comm.universe);

	if (rc != MPI_SUCCESS || g.comm.universe == MPI_COMM_NULL) {

		log_mpi_error("MPI_Comm_create_from_group(universe)", rc);
		std::abort();
	}

	rc = MPI_Comm_dup(g.comm.universe, &g.comm.app_universe);

	if (rc != MPI_SUCCESS || g.comm.app_universe == MPI_COMM_NULL) {

		log_mpi_error("MPI_Comm_dup(app_universe)", rc);
		std::abort();
	}

	rc = MPI_Comm_set_errhandler(g.comm.universe, MPI_ERRORS_RETURN);

	if (rc != MPI_SUCCESS) {

		log_mpi_error("MPI_Comm_set_errhandler(universe)", rc);
		std::abort();
	}

	rc = MPI_Comm_rank(g.comm.universe, &g.comm.u_rank);
	log_mpi_error("MPI_Comm_rank(universe)", rc);
	rc = MPI_Comm_size(g.comm.universe, &g.comm.u_size);
	log_mpi_error("MPI_Comm_size(universe)", rc);

	MAL_TRACE_START();
	if (g.comm.u_rank == 0) {
		MAL_TRACE_META("universe_size", g.comm.u_size);
	}

	if (g.cfg.resize_policy != MAL_RESIZE_POLICY_CUSTOM && g.cfg.decide_resize_func == nullptr) {

		switch (g.cfg.resize_policy) {

			case MAL_RESIZE_POLICY_AUTO: builtin_auto::install(0.6); break;
			case MAL_RESIZE_POLICY_THROUGHPUT: builtin_auto::install(0.0); break;
			case MAL_RESIZE_POLICY_EFFICIENCY: builtin_auto::install(0.8); break;
			case MAL_RESIZE_POLICY_FIXED_SEQUENCE: builtin_fixed_sequence::install(); break;
			case MAL_RESIZE_POLICY_COST: builtin_cost::install(); break;
			default: break;
		}
	}

	g.cfg.node_local_rank = detect_node_local_rank(g.comm.u_rank);

	#if defined(__linux__) || defined(__APPLE__)

	pin_main_thread_to_pcore();

	#endif

	int effective_initial_size = g.cfg.initial_size > 0 ? std::min(g.cfg.initial_size, g.comm.u_size) : g.comm.u_size;

	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		effective_initial_size = g.comm.u_size;
	}

	MAL_LOG_L(MAL_LOG_DEBUG, "CONFIG", "Initial active size: %d (universe=%d)", effective_initial_size, g.comm.u_size);
	if (g.comm.u_rank == 0) {
		MAL_TRACE_META("initial_size", effective_initial_size);
	}

	const bool init_immutable = !g.cfg.malleability_enabled.load(std::memory_order_relaxed) || (!g.cfg.enabled.load(std::memory_order_relaxed) && !g.cfg.load_balancing_enabled.load(std::memory_order_relaxed));

	if (init_immutable && effective_initial_size == g.comm.u_size) {

		g.comm.active = g.comm.universe;
		g.comm.active_borrowed = true;

	} else {

		int color = (g.comm.u_rank < effective_initial_size) ? 0 : MPI_UNDEFINED;
		rc = MPI_Comm_split(g.comm.universe, color, g.comm.u_rank, &g.comm.active);
		log_mpi_error("MPI_Comm_split(active, init)", rc);
	}

	if (g.comm.universe != MPI_COMM_NULL && g.comm.u_size > 1) {

		std::vector<char> warm_send(g.comm.u_size, 0), warm_recv(g.comm.u_size, 0);
		rc = MPI_Alltoall(warm_send.data(), 1, MPI_BYTE, warm_recv.data(), 1, MPI_BYTE, g.comm.universe);
		log_mpi_error("MPI_Alltoall(universe TCP warmup)", rc);
	}

	g.worker = std::thread(progress_thread);

	#if defined(__linux__)

	pthread_setname_np(g.worker.native_handle(), "mal_worker");

	#endif

	#if defined(__linux__) || defined(__APPLE__)

	pin_worker_thread_to_ecore(g.worker);

	#endif

	if (g.comm.active != MPI_COMM_NULL) {

		rc = MPI_Comm_set_errhandler(g.comm.active, MPI_ERRORS_RETURN);
		log_mpi_error("MPI_Comm_set_errhandler(active, init)", rc);

		rc = MPI_Comm_rank(g.comm.active, &g.comm.a_rank);
		log_mpi_error("MPI_Comm_rank(active)", rc);
		rc = MPI_Comm_size(g.comm.active, &g.comm.a_size);
		log_mpi_error("MPI_Comm_size(active)", rc);

	} else {

		g.comm.a_rank = -1;
		g.comm.a_size = 0;
	}

	g.timing.init = MPI_Wtime() - t_init_start;
}

void vec_scatter(MalVec& v, const void* root_data, const std::vector<long>& cuts) {

	std::vector<int> sc, sd;

	if (g.comm.a_rank == 0) {

		sc.resize(g.comm.a_size);

		const bool use_cuts = ((int)cuts.size() == g.comm.a_size + 1);

		for (int k = 0; k < g.comm.a_size; k++) {

			long ks, ke;

			if (use_cuts) {

				ks = cuts[(size_t)k];
				ke = cuts[(size_t)k + 1];

			} else {

				distribute(v.total_n, g.comm.a_size, k, ks, ke);
			}

			long bytes = (ke - ks) * (long)v.elem_size;

			if (MAL_UNLIKELY(bytes > INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "SCATTER", "Per-rank send size overflow (%ld bytes) for rank=%d in vec_scatter", bytes, k);
				MPI_Abort(g.comm.universe, 1);
			}

			sc[k] = (int)bytes;
		}

		sd = make_displs(sc);
	}

	long rc_bytes = v.local_n * (long)v.elem_size;

	if (MAL_UNLIKELY(rc_bytes > INT_MAX)) {

		MAL_LOG_L(MAL_LOG_ERROR, "SCATTER", "Local receive size overflow (%ld bytes) in vec_scatter", rc_bytes);
		MPI_Abort(g.comm.universe, 1);
	}

	int rc = (int)rc_bytes;

	MPI_Scatterv(root_data, g.comm.a_rank == 0 ? sc.data() : nullptr, g.comm.a_rank == 0 ? sd.data() : nullptr, MPI_BYTE, v.buf, rc, MPI_BYTE, 0, g.comm.active);
}

void vec_gather(MalVec& v) {

	if (v.total_n == 0) {

		return;
	}

	if (g.loop && !v.sealed) {

		const long confirmed = g.loop->confirmed_iter.load(std::memory_order_acquire);
		const long true_done = std::clamp(confirmed + 1 - v.buf_global_start, 0L, v.local_n);

		if (true_done > v.done_n) {

			append_done_segments(v, *g.loop, v.plan_origin_n, v.done_n, true_done);
			v.done_n = true_done;
		}

	} else if (v.local_n > v.done_n) {

		v.done_segs.push_back({v.buf_global_start + v.done_n, v.local_n - v.done_n});
	}

	const int usiz = g.comm.u_size;
	const bool root = (g.comm.u_rank == v.gather_root);

	std::vector<long> my_seg_flat;
	my_seg_flat.reserve(v.done_segs.size() * 2);

	for (auto [s, c] : v.done_segs) {

		my_seg_flat.push_back(s);
		my_seg_flat.push_back(c);
	}

	int my_seg_count = (int)my_seg_flat.size();

	std::vector<int> seg_counts, seg_displs;

	if (root) {

		seg_counts.resize(usiz);
	}

	MPI_Gather(&my_seg_count, 1, MPI_INT, root ? seg_counts.data() : nullptr, 1, MPI_INT, v.gather_root, g.comm.universe);

	std::vector<long> all_segs;

	if (root) {

		seg_displs = make_displs(seg_counts);
		all_segs.resize(seg_displs.back() + seg_counts.back());
	}

	MPI_Gatherv(my_seg_flat.empty() ? nullptr : my_seg_flat.data(), my_seg_count, MPI_LONG, all_segs.empty() ? nullptr : all_segs.data(), root ? seg_counts.data() : nullptr, root ? seg_displs.data() : nullptr, MPI_LONG, v.gather_root, g.comm.universe);

	long my_done_elems = 0;

	for (size_t si = 1; si < my_seg_flat.size(); si += 2) {

		my_done_elems += my_seg_flat[si];
	}

	if (MAL_UNLIKELY(my_done_elems < 0)) {

		MAL_LOG_L(MAL_LOG_ERROR, "GATHER", "Negative done element count in vec_gather (%ld)", my_done_elems);
		MPI_Abort(g.comm.universe, 1);
	}

	if (MAL_UNLIKELY(my_done_elems > v.local_n)) {

		MAL_LOG_L(MAL_LOG_WARN, "GATHER", "done segments exceed local buffer (%ld > %ld), clamping", my_done_elems, v.local_n);
		my_done_elems = v.local_n;
	}

	long my_data_bytes = my_done_elems * (long)v.elem_size;

	if (MAL_UNLIKELY(my_data_bytes > INT_MAX)) {

		MAL_LOG_L(MAL_LOG_ERROR, "GATHER", "Local send size overflow in vec_gather (%ld bytes)", my_data_bytes);
		MPI_Abort(g.comm.universe, 1);
	}

	std::vector<int> data_counts, data_displs;
	static thread_local void* tl_recv_raw = nullptr;
	static thread_local size_t tl_recv_cap = 0;
	void* recv_raw = nullptr;
	size_t recv_cap = 0;

	if (root) {

		data_counts.resize(usiz);

		for (int k = 0; k < usiz; k++) {

			long rank_elems = 0;

			for (int s = 0; s < seg_counts[k] / 2; s++) {

				rank_elems += all_segs[seg_displs[k] + s * 2 + 1];
			}

			long bytes_k = rank_elems * (long)v.elem_size;

			if (MAL_UNLIKELY(bytes_k < 0 || bytes_k > INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "GATHER", "Invalid computed data size from rank=%d (%ld bytes)", k, bytes_k);
				MPI_Abort(g.comm.universe, 1);
			}

			data_counts[k] = (int)bytes_k;
		}

		data_displs = make_displs(data_counts);

		size_t total_recv = (size_t)(data_displs.back() + data_counts.back());
		pool_reserve(tl_recv_raw, tl_recv_cap, total_recv > 0 ? total_recv : 1, false);
		recv_raw = tl_recv_raw;
		recv_cap = tl_recv_cap;
	}

	MPI_Gatherv(my_data_bytes > 0 ? v.buf : nullptr, (int)my_data_bytes, MPI_BYTE, recv_raw, root ? data_counts.data() : nullptr, root ? data_displs.data() : nullptr, MPI_BYTE, v.gather_root, g.comm.universe);

	if (root && v.result_buf) {

		char* recv_buf = static_cast<char*>(recv_raw);

		for (int k = 0; k < usiz; k++) {

			long rank_used = 0;
			const long rank_bytes = data_counts[k];

			for (int s = 0; s < seg_counts[k] / 2; s++) {

				long gs = all_segs[seg_displs[k] + s * 2];
				long cnt = all_segs[seg_displs[k] + s * 2 + 1];
				const long seg_bytes = cnt * (long)v.elem_size;

				if (seg_bytes <= 0) {

					continue;
				}

				if (rank_used >= rank_bytes) {

					MAL_LOG_L(MAL_LOG_WARN, "GATHER", "Insufficient gathered bytes for rank=%d (segments exceed payload)", k);
					break;
				}

				const long available = rank_bytes - rank_used;
				const long copy_bytes = std::min(seg_bytes, available);

				std::memcpy(static_cast<char*>(v.result_buf) + gs * (long)v.elem_size, recv_buf + data_displs[k] + rank_used, copy_bytes);

				if (copy_bytes < seg_bytes) {

					MAL_LOG_L(MAL_LOG_WARN, "GATHER", "Truncated segment copy for rank=%d seg=%d (%ld/%ld bytes)", k, s, copy_bytes, seg_bytes);
					break;
				}

				rank_used += copy_bytes;
			}
		}
	}

	if (recv_raw) {

		tl_recv_raw = recv_raw;
		tl_recv_cap = recv_cap;
	}
}

void mal_loop_horizon(long steps_remaining) {

	g.sync.iter_horizon.store(steps_remaining > 1 ? steps_remaining : 1, std::memory_order_release);

	if (steps_remaining > 1) {

		g.sync.iterative_kernel.store(true, std::memory_order_release);
	}
}

void mal_allgather_replicated(MalFor& f, void* full_buf, size_t elem_size, long total_n) {

	if (g.comm.active == MPI_COMM_NULL || g.comm.a_size <= 1 || total_n <= 0) {

		return;
	}

	const int asz = g.comm.a_size;

	long me[2] = {f.start, f.end - f.start};
	std::vector<long> all((size_t)asz * 2);
	int rc = MPI_Allgather(me, 2, MPI_LONG, all.data(), 2, MPI_LONG, g.comm.active);
	log_mpi_error("MPI_Allgather(allgather_replicated meta)", rc);

	std::vector<int> cb((size_t)asz), db((size_t)asz);
	bool overflow = false;

	for (int r = 0; r < asz; r++) {

		const long st = all[(size_t)r * 2];
		const long cn = all[(size_t)r * 2 + 1];
		const long cbytes = cn * (long)elem_size;
		const long dbytes = st * (long)elem_size;

		if (cbytes > INT_MAX || dbytes > INT_MAX) {

			overflow = true;
		}

		cb[(size_t)r] = (int)cbytes;
		db[(size_t)r] = (int)dbytes;
	}

	if (overflow) {

		MAL_LOG_L(MAL_LOG_ERROR, "ALLGATHER", "field too large for MPI int counts (total_n=%ld elem=%zu)", total_n, elem_size);
		MPI_Abort(g.comm.active, 1);
		return;
	}

	rc = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, full_buf, cb.data(), db.data(), MPI_BYTE, g.comm.active);
	log_mpi_error("MPI_Allgatherv(allgather_replicated)", rc);
}

void mal_sync_impl(MalFor& f, void* buf, int count, MPI_Datatype dtype, MPI_Op op) {

	if (count <= 0) {

		return;
	}

	if (count == 1) {

		for (MalAcc* a : f.accs) {

			if (a && a->ptr == buf && !a->sealed) {

				if (g.comm.app_universe != MPI_COMM_NULL && g.comm.u_size > 1) {

					int rc = MPI_Allreduce(MPI_IN_PLACE, buf, 1, dtype, op, g.comm.app_universe);
					log_mpi_error("MPI_Allreduce(mal_sync acc)", rc);
				}

				if (!a->epoch_buf.empty()) {

					combine_with_op(buf, a->epoch_buf.data(), a->dtype_idx, a->dop_idx);
				}

				a->sealed = true;
				return;
			}
		}
	}

	if (g.comm.active == MPI_COMM_NULL || g.comm.a_size <= 1) {

		return;
	}

	int rc = MPI_Allreduce(MPI_IN_PLACE, buf, count, dtype, op, g.comm.active);
	log_mpi_error("MPI_Allreduce(mal_sync)", rc);
}

void mal_bcast_impl(void* buf, int count, MPI_Datatype dtype, int root) {

	if (g.comm.active == MPI_COMM_NULL || g.comm.a_size <= 1 || count <= 0) {

		return;
	}

	int rc = MPI_Bcast(buf, count, dtype, root, g.comm.active);
	log_mpi_error("MPI_Bcast(mal_bcast)", rc);
}

void halo_exchange_field(MalFor& f, void* buf, size_t elem, long total) {

	if (g.comm.active == MPI_COMM_NULL || g.comm.a_size <= 1 || total <= 0) {

		return;
	}

	const long start = f.start;
	const long end = f.end;

	if (start >= end) {

		return;
	}

	const int n = g.comm.a_size;
	const int a = g.comm.a_rank;
	const int left = (a - 1 + n) % n;
	const int right = (a + 1) % n;

	char* b = static_cast<char*>(buf);
	const long gl = (start - 1 + total) % total;
	const long gr = end % total;
	const int es = (int)elem;

	int rc = MPI_Sendrecv(b + start * elem, es, MPI_BYTE, left, 0, b + gr * elem, es, MPI_BYTE, right, 0, g.comm.active, MPI_STATUS_IGNORE);
	log_mpi_error("MPI_Sendrecv(halo left)", rc);
	rc = MPI_Sendrecv(b + (end - 1) * elem, es, MPI_BYTE, right, 1, b + gl * elem, es, MPI_BYTE, left, 1, g.comm.active, MPI_STATUS_IGNORE);
	log_mpi_error("MPI_Sendrecv(halo right)", rc);
}

void mal_step_sync(MalFor& f, void* full_buf, size_t elem_size, long total_n) {

	const bool active_set_immutable = !g.cfg.malleability_enabled.load(std::memory_order_relaxed) || (!g.cfg.enabled.load(std::memory_order_relaxed) && !g.cfg.load_balancing_enabled.load(std::memory_order_relaxed));

	if (active_set_immutable) {

		if (g.sync.iter_horizon.load(std::memory_order_acquire) <= 1) {

			mal_allgather_replicated(f, full_buf, elem_size, total_n);

		} else {

			halo_exchange_field(f, full_buf, elem_size, total_n);
		}

		return;
	}

	if (g.sync.stop.load(std::memory_order_acquire)) {

		return;
	}

	f.confirmed_iter.store(f.end, std::memory_order_release);

	g.sync.step_buf = full_buf;
	g.sync.step_elem = elem_size;
	g.sync.step_total_n = total_n;
	g.sync.step_done.store(false, std::memory_order_release);
	g.sync.step_request.store(true, std::memory_order_release);
	g.sync.notify();

	g.sync.compute_wait([] {
		return g.sync.step_done.load(std::memory_order_acquire) || g.sync.stop.load(std::memory_order_acquire);
	});
}

void mal_step(MalFor& f, void* full_buf, size_t elem_size, long total_n) {

	const bool active_set_immutable = !g.cfg.malleability_enabled.load(std::memory_order_relaxed) || (!g.cfg.enabled.load(std::memory_order_relaxed) && !g.cfg.load_balancing_enabled.load(std::memory_order_relaxed));

	if (active_set_immutable) {

		mal_step_sync(f, full_buf, elem_size, total_n);
		return;
	}

	if (g.sync.stop.load(std::memory_order_acquire)) {

		return;
	}

	const long long step_no = g.sync.step_counter.fetch_add(1, std::memory_order_acq_rel) + 1;
	long long k = g.sync.eval_stride.load(std::memory_order_acquire);

	if (k <= 0) {

		k = g.cfg.stencil_epoch_steps.load(std::memory_order_relaxed);
	}

	if (k < 1) {

		k = 1;
	}

	const bool last_step = g.sync.iter_horizon.load(std::memory_order_acquire) <= 1;
	const bool finalize_local = g.sync.finalize_requested.load(std::memory_order_acquire);
	const bool eval_step = last_step || finalize_local || (step_no % (long long)k == 0);

	if (!eval_step) {

		halo_exchange_field(f, full_buf, elem_size, total_n);

		if (g.comm.active != MPI_COMM_NULL && g.comm.a_size > 1 && f.end > f.start) {

			const double* b = static_cast<const double*>(full_buf);
			double local_res = 0.0;

			if (elem_size == sizeof(double)) {

				for (long i = f.start; i < f.end; i++) {

					local_res += b[(size_t)i] * b[(size_t)i];
				}
			}

			int reduces = g.cfg.stencil_resid_reduces.load(std::memory_order_relaxed);

			if (reduces < 1) {

				reduces = 1;
			}

			for (int rr = 0; rr < reduces; rr++) {

				double res = 0.0;
				int rc = MPI_Allreduce(&local_res, &res, 1, MPI_DOUBLE, MPI_SUM, g.comm.active);
				log_mpi_error("MPI_Allreduce(mal_step residual)", rc);
				local_res = res / (double)g.comm.a_size;
			}
		}

		return;
	}

	g.sync.step_force_eval.store(true, std::memory_order_release);
	mal_step_sync(f, full_buf, elem_size, total_n);
	g.sync.step_force_eval.store(false, std::memory_order_release);
}

void mal_finalize() {

	const int saved_u_rank = g.comm.u_rank;

	mal_wait_attach_tasks();

	g.sync.finalize_requested.store(true, std::memory_order_release);

	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		g.sync.stop.store(true, std::memory_order_release);
	}

	g.sync.notify();

	double t0 = MPI_Wtime();
	g.worker.join();
	g.timing.finalize_worker_join = MPI_Wtime() - t0;

	t0 = MPI_Wtime();
	int rc;

	if (g.comm.universe != MPI_COMM_NULL) {

		rc = MPI_Barrier(g.comm.universe);
		log_mpi_error("MPI_Barrier(finalize entry)", rc);
	}

	std::vector<int> local_gather_roots;
	std::vector<MalVec*> local_gather_vecs;

	for (auto& vp : g.vecs) {

		if (vp->gather_root >= 0) {

			local_gather_roots.push_back(vp->gather_root);
			local_gather_vecs.push_back(vp.get());
		}
	}

	const bool active_was_immutable = !g.cfg.malleability_enabled.load(std::memory_order_relaxed) || (!g.cfg.enabled.load(std::memory_order_relaxed) && !g.cfg.load_balancing_enabled.load(std::memory_order_relaxed));
	const bool reconcile = !active_was_immutable && (g.timing.resize_count > 0);

	int local_n_gathers = (int)local_gather_roots.size();
	int n_gathers = local_n_gathers;

	if (reconcile) {

		MPI_Allreduce(&local_n_gathers, &n_gathers, 1, MPI_INT, MPI_MAX, g.comm.universe);
	}

	if (n_gathers > 0) {

		std::vector<int> roots = local_gather_roots;

		if (reconcile) {

			roots.assign(n_gathers, 0);
			int src_candidate = (local_n_gathers == n_gathers) ? g.comm.u_rank : g.comm.u_size;
			int bcast_src = 0;
			MPI_Allreduce(&src_candidate, &bcast_src, 1, MPI_INT, MPI_MIN, g.comm.universe);

			if (g.comm.u_rank == bcast_src) {

				roots = local_gather_roots;
			}

			MPI_Bcast(roots.data(), n_gathers, MPI_INT, bcast_src, g.comm.universe);
		}

		for (int gi = 0; gi < n_gathers; gi++) {

			if (gi < local_n_gathers) {

				vec_gather(*local_gather_vecs[(size_t)gi]);

			} else {

				int gather_root = roots[gi];
				int zero = 0;
				MPI_Gather(&zero, 1, MPI_INT, nullptr, 1, MPI_INT, gather_root, g.comm.universe);
				MPI_Gatherv(nullptr, 0, MPI_LONG, nullptr, nullptr, nullptr, MPI_LONG, gather_root, g.comm.universe);
				MPI_Gatherv(nullptr, 0, MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE, gather_root, g.comm.universe);
			}
		}
	}

	for (auto& vp : g.vecs) {

		MalVec& v = *vp;

		if (v.user_ptr) {

			*v.user_ptr = (g.comm.u_rank == v.gather_root && v.result_buf) ? v.result_buf : nullptr;
		}

		if (v.result_buf && !v.user_ptr) {

			std::free(v.result_buf);
			v.result_buf = nullptr;
		}

		v.free_resources();
	}

	g.vecs.clear();
	g.timing.finalize_vec_gather = MPI_Wtime() - t0;

	t0 = MPI_Wtime();
	int naccs = (int)g.accs.size();
	int local_naccs = naccs;

	if (reconcile) {

		rc = MPI_Bcast(&naccs, 1, MPI_INT, 0, g.comm.universe);
		log_mpi_error("MPI_Bcast(naccs, finalize)", rc);
	}

	if (MAL_UNLIKELY(naccs != local_naccs)) {

		MAL_LOG_L(MAL_LOG_WARN, "FINALIZE", "Acc count mismatch (local=%d root=%d); reducing identity for missing entries", local_naccs, naccs);
	}

	struct FinalAccGetter {

		MalAcc* operator()(int k) const {

			return (k >= 0 && (size_t)k < g.accs.size()) ? g.accs[(size_t)k].get() : nullptr;
		}
	};

	struct FinalAccSetter {

		void operator()(int k, const char* r, int) const {

			if (k < 0 || (size_t)k >= g.accs.size()) {

				return;
			}

			MalAcc* a = g.accs[(size_t)k].get();

			if (!a || !a->ptr || a->sealed) {

				return;
			}

			if (g.comm.u_rank == a->result_rank) {

				a->fn_set(a->ptr, r);

			} else {

				write_identity(static_cast<char*>(a->ptr), a->dtype_idx, a->dop_idx, (int)a->esz);
			}
		}
	};

	batched_allreduce(naccs, FinalAccGetter{}, FinalAccSetter{});

	g.accs.clear();

	if (g.loop) {

		g.loop->vecs.clear();
		g.loop->accs.clear();
		g.loop->phase.store(MAL_LOOP_FINISHED, std::memory_order_release);
		g.loop = nullptr;
	}

	g.timing.finalize_acc_reduce = MPI_Wtime() - t0;

	t0 = MPI_Wtime();

	for (auto& sp : g.shared) {

		sp->free_resources();
	}

	g.shared.clear();

	if (g.pending) {

		g.pending.reset();
	}

	for (auto& e : g.gather_cache) {

		g_buffer_pool.release(e.ptr, e.bytes);
	}

	g.gather_cache.clear();

	if (g.comm.universe != MPI_COMM_NULL) {

		rc = MPI_Barrier(g.comm.universe);
		log_mpi_error("MPI_Barrier(finalize teardown)", rc);
	}

	if (g.comm.active != MPI_COMM_NULL && !g.comm.active_borrowed) {

		rc = MPI_Comm_free(&g.comm.active);
		log_mpi_error("MPI_Comm_free(active)", rc);
	}

	g.comm.active = MPI_COMM_NULL;

	rc = MPI_Group_free(&g.comm.world_group);
	log_mpi_error("MPI_Group_free(world_group)", rc);
	g.comm.world_group = MPI_GROUP_NULL;

	rc = MPI_Comm_free(&g.comm.app_universe);
	log_mpi_error("MPI_Comm_free(app_universe)", rc);
	g.comm.app_universe = MPI_COMM_NULL;

	rc = MPI_Comm_free(&g.comm.universe);
	log_mpi_error("MPI_Comm_free(universe)", rc);
	g.comm.universe = MPI_COMM_NULL;

	if (g.cfg.decide_resize_plugin_handle) {
		dlclose(g.cfg.decide_resize_plugin_handle);
		g.cfg.decide_resize_plugin_handle = nullptr;
	}

	rc = MPI_Session_finalize(&g.comm.session);
	log_mpi_error("MPI_Session_finalize(session)", rc);
	g.comm.session = MPI_SESSION_NULL;

	g.timing.finalize_cleanup = MPI_Wtime() - t0;

	if (g.timing.enabled) {

		mal_trace_timer(saved_u_rank, "init", g.timing.init);
		mal_trace_timer(saved_u_rank, "mal_for", g.timing.mal_for_total);
		mal_trace_timer(saved_u_rank, "attach", g.timing.attach_total);
		mal_trace_timer(saved_u_rank, "check_wait_resize", g.timing.check_wait_resize);
		mal_trace_timer(saved_u_rank, "check_wait_attach", g.timing.check_wait_attach);
		mal_trace_timer(saved_u_rank, "check_wait_total", g.timing.check_wait_total);
		mal_trace_timer(saved_u_rank, "check_for_total", g.timing.check_for_total);
		mal_trace_timer(saved_u_rank, "resize_prepare", g.timing.resize_prepare);
		mal_trace_timer(saved_u_rank, "resize_commit", g.timing.resize_commit);
		mal_trace_timer(saved_u_rank, "resize_count", g.timing.resize_count);
		mal_trace_timer(saved_u_rank, "epoch_decision", g.timing.epoch_decision);
		mal_trace_timer(saved_u_rank, "epoch_decision_count", g.timing.epoch_decision_count);
		mal_trace_timer(saved_u_rank, "fin_worker_join", g.timing.finalize_worker_join);
		mal_trace_timer(saved_u_rank, "fin_vec_gather", g.timing.finalize_vec_gather);
		mal_trace_timer(saved_u_rank, "fin_acc_reduce", g.timing.finalize_acc_reduce);
		mal_trace_timer(saved_u_rank, "fin_cleanup", g.timing.finalize_cleanup);
		mal_trace_timer(saved_u_rank, "wait_for_compute", g.timing.wait_for_compute);
		std::fflush(stdout);
	}

	MAL_TRACE_END();
}

inline bool can_enter_running_phase(const MalFor& f, bool ignore_attach_pending_gate = false) {

	if (g.sync.stop.load(std::memory_order_relaxed) || f.start >= f.end) {

		return false;
	}

	return ignore_attach_pending_gate || !g.sync.attach_pending.load(std::memory_order_relaxed);
}

inline void maybe_enter_running_phase(MalFor& f, bool ignore_attach_pending_gate = false) {

	if (can_enter_running_phase(f, ignore_attach_pending_gate)) {

		f.phase.store(MAL_LOOP_RUNNING, std::memory_order_relaxed);
	}
}

void seal_loop_vecs(MalFor& prev) {

	const long confirmed = prev.confirmed_iter.load(std::memory_order_acquire);

	for (MalVec* v : prev.vecs) {

		if (v == nullptr || v->sealed) {

			continue;
		}

		if (v->total_n > 0) {

			const long true_done = std::clamp(confirmed + 1 - v->buf_global_start, 0L, v->local_n);

			if (true_done > v->done_n) {

				append_done_segments(*v, prev, v->plan_origin_n, v->done_n, true_done);
				v->done_n = true_done;
			}
		}

		v->sealed = true;
	}
}

MalFor mal_for(long total_iters, long* iter, long* limit) {

	const double t_for_start = MPI_Wtime();

	MalFor f;

	f.user_iter = iter;
	f.user_limit = limit;
	f.gen = g.loop_gen.fetch_add(1, std::memory_order_acq_rel) + 1;

	if (g.loop != nullptr && g.loop != &f) {

		seal_loop_vecs(*g.loop);
	}

	if (g.sync.resize_pending.load(std::memory_order_acquire) && !g.sync.stop.load(std::memory_order_acquire)) {

		g.sync.compute_wait(resize_pending_cleared_or_stop);
	}

	if (g.sync.pending_has_ranges.load(std::memory_order_acquire)) {

		load_pending_ranges_into_loop(f);
		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	} else if (g.comm.a_size > 0 && !g.sync.stop.load(std::memory_order_relaxed)) {

		weighted_distribute(total_iters, g.comm.a_size, g.comm.a_rank, f.start, f.end);
		install_loop_plan(f, {{f.start, f.end}});

		if (g.sync.iterative_kernel.load(std::memory_order_acquire)) {

			if (g.lb.epoch_start_time <= 0.0) {

				g.lb.epoch_start_time = MPI_Wtime();
				g.lb.epoch_assigned = 0;
			}

			g.lb.epoch_assigned += f.end - f.start;

		} else {

			g.lb.epoch_assigned = f.end - f.start;
			g.lb.epoch_start_time = MPI_Wtime();
		}

		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	} else {

		f.start = f.end = 0;
		f.plan_ranges.clear();
		f.plan_local_bases.clear();

		f.phase.store(MAL_LOOP_WAITING_ACTIVATION, std::memory_order_relaxed);
	}

	set_iter(f, f.start);
	*limit = f.end;

	f.confirmed_iter.store(f.start - 1, std::memory_order_release);

	{

		std::lock_guard lk(g.sync.mu);
		g.sync.compute_ready.store(false, std::memory_order_release);
	}

	g.loop = &f;

	const bool iterative_loop = g.sync.iterative_kernel.load(std::memory_order_acquire);

	const bool skip_idle_activation_wait = (f.start == f.end) && !g.sync.pending_has_ranges.load(std::memory_order_acquire) && (iterative_loop || total_iters <= g.comm.a_size);

	while (!skip_idle_activation_wait && f.start == f.end && !g.sync.stop.load(std::memory_order_acquire) && g.sync.loop_done_gen.load(std::memory_order_acquire) < f.gen) {

		f.current = f.end;
		f.phase.store(MAL_LOOP_WAITING_ACTIVATION, std::memory_order_relaxed);
		g.sync.compute_wait([&f] {
			return g.sync.stop.load(std::memory_order_acquire) || g.sync.pending_has_ranges.load(std::memory_order_acquire) || g.sync.loop_has_new_work.load(std::memory_order_acquire) || g.sync.loop_done_gen.load(std::memory_order_acquire) >= f.gen;
		});

		if (g.sync.stop.load(std::memory_order_acquire) || g.sync.loop_done_gen.load(std::memory_order_acquire) >= f.gen) {

			break;
		}

		if (f.start == f.end && g.sync.pending_has_ranges.load(std::memory_order_acquire)) {

			load_pending_ranges_into_loop(f);
			f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);
		}
	}

	maybe_enter_running_phase(f);

	g.timing.mal_for_total += MPI_Wtime() - t_for_start;

	return f;
}

void advance_next_range(MalFor& f) {

	long a, b;

	{

		std::lock_guard lk(g.sync.plan_mu);

		f.plan_idx++;
		a = f.plan_ranges[f.plan_idx].first;
		b = f.plan_ranges[f.plan_idx].second;

		f.start = a;

		sync_vec_mapping_for_current_range(f);

		set_limit(f, b);

		prime_range_start(f);
	}

	MAL_LOG_L(MAL_LOG_DEBUG, "RANGE", "Next range [%ld, %ld) (base=%ld)", a, b, current_range_local_base(f));
}

void mal_check_for(MalFor& f) {

	const bool timing_enabled = MAL_UNLIKELY(g.timing.enabled);
	const double t0 = timing_enabled ? MPI_Wtime() : 0.0;
	bool waited = false;

	struct ConfirmedExitGuard {

		MalFor* f;

		~ConfirmedExitGuard() {

			if (!f->accs.empty()) {

				std::lock_guard lk(g.sync.plan_mu);

				for (MalAcc* a : f->accs) {

					if (a && !a->sealed) {

						if (a->shadow.size() < a->esz) {

							a->shadow.resize(a->esz);
						}

						a->fn_get(a->ptr, a->shadow.data());
						a->shadow_iter = *f->user_iter;
					}
				}

				f->confirmed_iter.store(*f->user_iter, std::memory_order_release);
				return;
			}

			f->confirmed_iter.store(*f->user_iter, std::memory_order_release);
		}

	} confirmed_guard{&f};

	g.sync.compute_epoch.store(f.check_counter++, std::memory_order_relaxed);

	f.current = *f.user_iter;
	const bool attach_pending = g.sync.attach_pending.load(std::memory_order_relaxed);

	if (MAL_UNLIKELY(attach_pending)) {

		const double t1 = timing_enabled ? MPI_Wtime() : 0.0;
		f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

		g.sync.compute_wait(attach_pending_cleared_or_stop);

		waited = true;

		if (timing_enabled) {

			const double dt = MPI_Wtime() - t1;
			g.timing.check_wait_attach += dt;
			g.timing.check_wait_total += dt;
		}

		maybe_enter_running_phase(f, true);
	}

	const bool resize_pending = g.sync.resize_pending.load(std::memory_order_relaxed);

	if (MAL_UNLIKELY(resize_pending)) {

		const double t1 = timing_enabled ? MPI_Wtime() : 0.0;
		g.sync.compute_wait(resize_pending_cleared_or_stop);

		waited = true;

		if (timing_enabled) {

			const double dt = MPI_Wtime() - t1;
			g.timing.check_wait_resize += dt;
			g.timing.check_wait_total += dt;
		}

		if (g.comm.active != MPI_COMM_NULL && f.start == f.end && g.pending && !g.pending->ranges.empty()) {

			load_pending_ranges_into_loop(f);
			prime_range_start(f);
			return;
		}

		{

			const bool had_new_work = g.sync.loop_has_new_work.load(std::memory_order_relaxed);
			g.sync.loop_has_new_work.store(false, std::memory_order_relaxed);

			if (g.comm.active != MPI_COMM_NULL && f.start < f.end) {

				if (had_new_work) {

					prime_range_start(f);
				}

				return;
			}
		}
	}

	if (MAL_LIKELY(f.current + 1 < f.end)) {

		return;
	}

	if (f.plan_idx + 1 < f.plan_ranges.size()) {

		advance_next_range(f);

		return;
	}

	if (g.sync.stop) {

		return;
	}

	const bool active_set_immutable = !g.cfg.malleability_enabled.load(std::memory_order_relaxed) || (!g.cfg.enabled.load(std::memory_order_relaxed) && !g.cfg.load_balancing_enabled.load(std::memory_order_relaxed));

	if (active_set_immutable) {

		return;
	}

	f.current = f.end;

	while (!g.sync.stop.load(std::memory_order_acquire)) {

		if (g.sync.loop_done_gen.load(std::memory_order_acquire) >= f.gen) {

			break;
		}

		const bool has_pending = g.sync.pending_has_ranges.load(std::memory_order_acquire);
		const bool has_resize_pending = g.sync.resize_pending.load(std::memory_order_acquire);
		const bool has_attach = g.sync.attach_pending.load(std::memory_order_acquire);
		const bool has_new_work = g.sync.loop_has_new_work.load(std::memory_order_acquire);

		if (!has_pending && !has_resize_pending && !has_attach && !has_new_work) {

			if (g.sync.stop.load(std::memory_order_acquire)) {

				break;
			}

			const double t1 = timing_enabled ? MPI_Wtime() : 0.0;
			g.sync.compute_wait([&] {
				return g.sync.stop.load(std::memory_order_acquire) || g.sync.resize_pending.load(std::memory_order_acquire) || g.sync.pending_has_ranges.load(std::memory_order_acquire) || g.sync.loop_has_new_work.load(std::memory_order_acquire) || g.sync.loop_done_gen.load(std::memory_order_acquire) >= f.gen;
			});

			waited = true;

			if (timing_enabled) {

				g.timing.check_wait_total += MPI_Wtime() - t1;
			}

			if (g.sync.stop.load(std::memory_order_acquire)) {

				break;
			}
		}

		{

			const double t1 = timing_enabled ? MPI_Wtime() : 0.0;
			g.sync.compute_wait(resize_pending_cleared_or_stop);
			waited = true;

			if (timing_enabled) {

				g.timing.check_wait_total += MPI_Wtime() - t1;
			}
		}

		if (g.sync.stop.load(std::memory_order_acquire)) {

			break;
		}

		if (g.sync.pending_has_ranges.load(std::memory_order_acquire)) {

			if (g.comm.active != MPI_COMM_NULL && f.start == f.end && g.pending && !g.pending->ranges.empty()) {

				load_pending_ranges_into_loop(f);
			}

			if (f.start < f.end) {

				prime_range_start(f);
				maybe_enter_running_phase(f);
				break;
			}
		}

		{

			const bool had_new_work = g.sync.loop_has_new_work.load(std::memory_order_relaxed);
			g.sync.loop_has_new_work.store(false, std::memory_order_relaxed);

			if (had_new_work && f.start < f.end) {

				prime_range_start(f);
				break;
			}
		}

		if (f.start < f.end) {

			break;
		}
	}

	if (timing_enabled && waited) {

		g.timing.check_for_total += MPI_Wtime() - t0;
	}
}

void mal_attach_vec(MalFor& f, void** user_ptr, size_t elem_size, long total_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	const double t_attach_start = MPI_Wtime();

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	auto vp = std::make_unique<MalVec>();
	MalVec* v = vp.get();

	void* orig = user_ptr ? *user_ptr : nullptr;
	long n = f.end - f.start;
	long planned_total = total_range_iters(f.plan_ranges);

	if (planned_total > 0) {

		n = planned_total;
	}

	std::vector<long> partition_cuts;

	if (policy == MAL_ATTACH_PARTITIONED && g.comm.a_size > 0 && g.comm.a_rank >= 0) {

		partition_cuts = build_partition_cuts(total_n, g.comm.a_size);
	}

	v->elem_size = elem_size;
	v->local_n = n;
	v->buf_global_start = f.start;
	v->total_n = total_n;
	v->user_ptr = user_ptr;
	v->gather_root = result_rank;
	v->attach_policy = policy;
	v->access_mode = access_mode;
	v->cache_valid = false;

	if (result_rank >= 0 && g.comm.u_rank == result_rank) {

		v->result_buf = orig ? orig : checked_realloc(nullptr, total_n > 0 ? (size_t)total_n * elem_size : 1, "mal_attach_vec.result_buf");
	}

	const bool shared_active = (policy == MAL_ATTACH_SHARED_ACTIVE);
	bool once_all = (policy == MAL_ATTACH_SHARED_ALL);
	const bool async_attach = use_async_attach_mode(exec_mode);

	if (MAL_UNLIKELY(elem_size == 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with elem_size=0");
	}

	if (MAL_UNLIKELY(total_n < 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_vec called with negative total_n=%ld", total_n);
	}

	if (MAL_UNLIKELY(elem_size > 0 && total_n > 0 && (size_t)total_n > SIZE_MAX / elem_size)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_vec total bytes overflow total_n=%ld elem_size=%zu", total_n, elem_size);
		MPI_Abort(g.comm.universe, 1);
	}

	if ((once_all || shared_active) && result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "Shared vector policy ignores gather result_rank=%d", result_rank);

		if (v->result_buf != nullptr && v->result_buf != orig) {

			std::free(v->result_buf);
		}

		v->result_buf = nullptr;
		v->gather_root = -1;
		once_all = false;
		result_rank = -1;
	}

	if (once_all || shared_active) {

		n = total_n;
		v->local_n = n;
		v->done_n = 0;
		v->buf_global_start = 0;
	}

	v->buf = static_cast<char*>(g_buffer_pool.acquire((v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size));
	v->buf_bytes = (v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size;
	v->plan_origin_n = v->done_n;

	int idx = (int)f.vecs.size();

	if (g.pending) {

		if (shared_active || once_all) {

			StagedBuffer staged = take_pending_shared_vec();

			if (staged.ptr) {

				const size_t copy_bytes = (size_t)std::max(0L, total_n) * elem_size;

				if (copy_bytes > 0 && staged.bytes >= copy_bytes) {

					g_buffer_pool.release(v->buf, v->buf_bytes);
					v->buf = static_cast<char*>(staged.ptr);
					v->buf_bytes = staged.bytes;

				} else {

					if (copy_bytes > 0) {

						std::memcpy(v->buf, staged.ptr, copy_bytes);
					}

					g_buffer_pool.release(staged.ptr, staged.bytes);
				}
			}

		} else {

			StagedBuffer stash = take_pending_vec_slice(idx);

			if (stash.ptr) {

				const size_t copy_bytes = (size_t)std::max(0L, n) * elem_size;

				if (copy_bytes > 0 && stash.bytes >= copy_bytes) {

					g_buffer_pool.release(v->buf, v->buf_bytes);
					v->buf = static_cast<char*>(stash.ptr);
					v->buf_bytes = stash.bytes;

				} else {

					if (copy_bytes > 0) {

						std::memcpy(v->buf, stash.ptr, copy_bytes);
					}

					g_buffer_pool.release(stash.ptr, stash.bytes);
				}
			}
		}
	}

	const bool inactive_no_pending = (g.comm.active == MPI_COMM_NULL && !g.pending && policy == MAL_ATTACH_PARTITIONED);

	if (inactive_no_pending && user_ptr) {

		*user_ptr = nullptr;

	} else {

		v->sync_user_ptr();
	}

	f.vecs.push_back(v);
	g.vecs.push_back(std::move(vp));

	const bool can_dispatch_attach = (!g.pending && g.comm.active != MPI_COMM_NULL);

	if (can_dispatch_attach) {

		const size_t total_bytes = (size_t)std::max(0L, total_n) * elem_size;

		if (once_all || shared_active) {

			run_shared_active_attach_bcast(*v, orig, total_bytes, exec_mode);

		} else {

			run_partitioned_attach_scatter(*v, orig, result_rank, total_bytes, exec_mode, std::move(partition_cuts));
		}
	}

	if (g.pending && idx + 1 == (int)g.pending->vec_slices.size()) {

		g.pending->vec_slices.clear();
	}

	maybe_enter_running_phase(f, !async_attach);

	g.timing.attach_total += MPI_Wtime() - t_attach_start;
}

void mal_attach_vec(MalForND& f, void** user_ptr, size_t elem_size, long total_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	mal_attach_vec(mal_for_nd_base(f), user_ptr, elem_size, total_n, result_rank, policy, exec_mode, access_mode);
}

void mal_attach_vec_ragged(MalFor& f, void** user_ptr, size_t elem_size, long total_inner, const long* row_offsets, long n_rows, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	const double t_attach_start = MPI_Wtime();
	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	if (MAL_UNLIKELY(!row_offsets || n_rows < 0)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_vec_ragged: null row_offsets or n_rows<0");
		return;
	}

	if (MAL_UNLIKELY(g.comm.a_size != g.comm.u_size)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_vec_ragged requires ALL ranks active (a_size=%d u_size=%d) — launch with MAL_START_AT_UNIVERSE=1 and MAL_INITIAL_SIZE=<universe>", g.comm.a_size, g.comm.u_size);
		MPI_Abort(g.comm.universe, 1);
	}

	auto vp = std::make_unique<MalVec>();
	MalVec* v = vp.get();
	void* orig = user_ptr ? *user_ptr : nullptr;

	if (g.comm.u_rank == 0 && orig && total_inner > 0) {

		v->ragged_full_bytes = (size_t)total_inner * elem_size;
		v->ragged_full_src = std::malloc(v->ragged_full_bytes);

		if (v->ragged_full_src) {

			std::memcpy(v->ragged_full_src, orig, v->ragged_full_bytes);

		} else {

			MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_vec_ragged: OOM retaining full source (%zu bytes)", v->ragged_full_bytes);
			MPI_Abort(g.comm.universe, 1);
		}
	}

	const long inner_start = row_offsets[f.start];
	const long inner_end = row_offsets[f.end];
	const long n = inner_end - inner_start;

	std::vector<long> inner_cuts;

	if (g.comm.a_size > 0 && g.comm.a_rank >= 0) {

		const std::vector<long> row_cuts = build_partition_cuts(n_rows, g.comm.a_size);
		inner_cuts.resize(row_cuts.size());

		for (size_t j = 0; j < row_cuts.size(); j++) {

			inner_cuts[j] = row_offsets[row_cuts[j]];
		}
	}

	v->elem_size = elem_size;
	v->local_n = n;
	v->buf_global_start = inner_start;
	v->total_n = total_inner;
	v->user_ptr = user_ptr;
	v->gather_root = -1;
	v->attach_policy = MAL_ATTACH_PARTITIONED;
	v->access_mode = access_mode;
	v->cache_valid = false;
	v->ragged = true;
	v->ragged_row_offsets = row_offsets;
	v->ragged_n_rows = n_rows;
	v->ragged_bases.assign(1, 0);

	v->buf = static_cast<char*>(g_buffer_pool.acquire((v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size));
	v->buf_bytes = (v->local_n > 0 ? (size_t)v->local_n : 1) * elem_size;
	v->plan_origin_n = v->done_n;

	const bool inactive_no_pending = (g.comm.active == MPI_COMM_NULL && !g.pending);

	if (inactive_no_pending && user_ptr) {

		*user_ptr = nullptr;

	} else {

		v->sync_user_ptr();
	}

	f.vecs.push_back(v);
	g.vecs.push_back(std::move(vp));

	const bool async_attach = use_async_attach_mode(exec_mode);
	const bool can_dispatch_attach = (!g.pending && g.comm.active != MPI_COMM_NULL);

	if (can_dispatch_attach) {

		const size_t total_bytes = (size_t)std::max(0L, total_inner) * elem_size;
		run_partitioned_attach_scatter(*v, orig, -1, total_bytes, exec_mode, std::move(inner_cuts));
	}

	maybe_enter_running_phase(f, !async_attach);

	g.timing.attach_total += MPI_Wtime() - t_attach_start;
}

void mal_attach_csr(MalFor& f, void** values, size_t value_elem_size, void** col_indices, size_t index_elem_size, long* row_ptr, long n_rows, long nnz) {

	if (MAL_UNLIKELY(!row_ptr || n_rows < 0)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_csr: null row_ptr or n_rows<0");
		return;
	}

	mal_bcast_impl(row_ptr, (int)(n_rows + 1), MPI_LONG, 0);

	mal_attach_vec_ragged(f, values, value_elem_size, nnz, row_ptr, n_rows);
	mal_attach_vec_ragged(f, col_indices, index_elem_size, nnz, row_ptr, n_rows);
}

void detail::acc_register(MalFor& f, detail::AccDesc d, int result_rank) {

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	auto ap = std::make_unique<MalAcc>();
	MalAcc* a = ap.get();

	a->result_rank = result_rank;
	a->ptr = d.ptr;
	a->dtype_idx = dtype_tag(d.dtype);
	a->dop_idx = dop_tag(d.dop);
	a->esz = d.esz;
	a->fn_get = d.fn_get;
	a->fn_set = d.fn_set;
	a->fn_add = d.fn_add;
	a->fn_reset = d.fn_reset;

	a->epoch_buf = take_pending_acc_epoch_buf(a->esz, a->dtype_idx, a->dop_idx);

	write_identity(static_cast<char*>(a->ptr), a->dtype_idx, a->dop_idx, (int)a->esz);

	f.accs.push_back(a);
	g.accs.push_back(std::move(ap));

	maybe_enter_running_phase(f, true);
}

void mal_attach_mat(MalFor& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	const double t_mat_start = MPI_Wtime();

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	if (MAL_UNLIKELY(elem_size == 0 || primary_n < 0 || secondary_n < 0)) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "mal_attach_mat called with invalid shape/size elem_size=%zu primary_n=%ld secondary_n=%ld", elem_size, primary_n, secondary_n);
	}

	if (policy == MAL_ATTACH_PARTITIONED) {

		mal_attach_vec(f, user_ptr, elem_size * (size_t)secondary_n, primary_n, result_rank, MAL_ATTACH_PARTITIONED, exec_mode, access_mode);

		return;
	}

	if (MAL_UNLIKELY(elem_size > 0 && secondary_n > 0 && (size_t)secondary_n > SIZE_MAX / elem_size)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_mat row stride overflow secondary_n=%ld elem_size=%zu", secondary_n, elem_size);
		MPI_Abort(g.comm.universe, 1);
	}

	const size_t row_bytes = (size_t)secondary_n * elem_size;

	if (MAL_UNLIKELY(row_bytes > 0 && primary_n > 0 && (size_t)primary_n > SIZE_MAX / row_bytes)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ATTACH", "mal_attach_mat total bytes overflow primary_n=%ld secondary_n=%ld elem_size=%zu", primary_n, secondary_n, elem_size);
		MPI_Abort(g.comm.universe, 1);
	}

	const size_t total_bytes = (size_t)primary_n * row_bytes;
	const bool shared_all = (policy == MAL_ATTACH_SHARED_ALL);
	const bool async_attach = use_async_attach_mode(exec_mode);

	if (!shared_all && g.comm.active == MPI_COMM_NULL) {

		if (user_ptr) {

			*user_ptr = nullptr;
		}

		auto sp = std::make_unique<SharedMat>();
		sp->buf = nullptr;
		sp->total_bytes = total_bytes;
		sp->user_owned = false;
		sp->user_ptr = user_ptr;

		g.shared.push_back(std::move(sp));

		maybe_enter_running_phase(f, !async_attach);

		g.timing.attach_total += MPI_Wtime() - t_mat_start;

		return;
	}

	if (result_rank >= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "result_rank=%d ignored for shared matrix policies", result_rank);
	}

	void* orig = user_ptr ? *user_ptr : nullptr;
	void* buf;

	if (g.pending) {

		StagedBuffer staged = take_pending_shared_mat();

		if (staged.ptr) {

			buf = staged.ptr;

		} else if (shared_all) {

			buf = g_buffer_pool.acquire(total_bytes > 0 ? total_bytes : 1);

		} else {

			buf = acquire_or_broadcast_active_shared_mat(orig, total_bytes, exec_mode);
		}

	} else {

		buf = acquire_or_broadcast_active_shared_mat(orig, total_bytes, exec_mode);
	}

	if (user_ptr) {

		*user_ptr = buf;
	}

	auto sp = std::make_unique<SharedMat>();
	sp->buf = buf;
	sp->total_bytes = total_bytes;
	sp->user_owned = (g.comm.a_rank == 0 && orig != nullptr);
	sp->user_ptr = user_ptr;

	g.shared.push_back(std::move(sp));

	maybe_enter_running_phase(f, !async_attach);

	g.timing.attach_total += MPI_Wtime() - t_mat_start;
}

void mal_attach_mat(MalForND& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank, MalAttachPolicy policy, MalAttachExecMode exec_mode, MalDataAccessMode access_mode) {

	mal_attach_mat(mal_for_nd_base(f), user_ptr, elem_size, primary_n, secondary_n, result_rank, policy, exec_mode, access_mode);
}
