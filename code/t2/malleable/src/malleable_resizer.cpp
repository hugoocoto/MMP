#ifndef MALLEABLE_RESIZER_CPP_INCLUDED
#define MALLEABLE_RESIZER_CPP_INCLUDED

#include "malleable_types.cpp"

constexpr double kEpsDone = 1.0;
constexpr double kEpsElapsed = 1e-6;
constexpr double kEpsWeight = 1e-12;
constexpr int kLbGatherFields = 2;
constexpr int kFusedGatherFields = 1 + kLbGatherFields;
constexpr int kMinResizeCooldownEpochs = 2;
constexpr double kImbHi = 1.50;
constexpr double kMinVoteTurnout = 0.5;

void mal_set_shared_mem(void* mem) {

	g.shared_mem.mem = mem;

}

void* mal_get_shared_mem() {

	return g.shared_mem.mem;

}

void mal_set_decide_resize_func(DecideResizeFunc func) {

	g.cfg.decide_resize_func = func;

}

static void* plugin_symbol_or_abort(void* handle, const char* name) {

	dlerror();
	void* sym = dlsym(handle, name);
	const char* err = dlerror();
	if (err || !sym) {
		MAL_LOG_L(MAL_LOG_ERROR, "PLUGIN", "dlsym(\"%s\") failed: %s", name, err ? err : "symbol is null");
		std::abort();
	}

	return sym;

}

void mal_set_decide_resize_plugin(const char* path, const char* func_name) {

	void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		MAL_LOG_L(MAL_LOG_ERROR, "PLUGIN", "dlopen(\"%s\") failed: %s", path, dlerror());
		std::abort();
	}

	void* sym = plugin_symbol_or_abort(handle, func_name);

	if (g.cfg.decide_resize_plugin_handle) {
		dlclose(g.cfg.decide_resize_plugin_handle);
	}
	g.cfg.decide_resize_plugin_handle = handle;

	DecideResizeFunc func;
	static_assert(sizeof(func) == sizeof(sym), "function pointer size mismatch");
	std::memcpy(&func, &sym, sizeof(func));
	mal_set_decide_resize_func(func);

}

void mal_set_decide_resize_state_funcs(ResizeStateSaveFunc save, ResizeStateLoadFunc load) {

	g.cfg.decide_resize_state_save = save;
	g.cfg.decide_resize_state_load = load;

}

void mal_set_decide_resize_state_plugin(const char* save_func_name, const char* load_func_name) {

	if (!g.cfg.decide_resize_plugin_handle) {
		MAL_LOG_L(MAL_LOG_ERROR, "PLUGIN", "mal_set_decide_resize_state_plugin() called before mal_set_decide_resize_plugin()");
		std::abort();
	}

	void* save_sym = plugin_symbol_or_abort(g.cfg.decide_resize_plugin_handle, save_func_name);
	void* load_sym = plugin_symbol_or_abort(g.cfg.decide_resize_plugin_handle, load_func_name);

	ResizeStateSaveFunc save;
	ResizeStateLoadFunc load;
	static_assert(sizeof(save) == sizeof(save_sym), "function pointer size mismatch");
	static_assert(sizeof(load) == sizeof(load_sym), "function pointer size mismatch");
	std::memcpy(&save, &save_sym, sizeof(save));
	std::memcpy(&load, &load_sym, sizeof(load));
	mal_set_decide_resize_state_funcs(save, load);

}

inline const double* lb_row_at(const std::vector<double>& all_lb_buf, int k) {

	return &all_lb_buf[(size_t)k * (size_t)kLbGatherFields];

}

inline double lb_row_tp(const double* row) {

	return (row[0] > 0.0 && row[1] > kEpsElapsed) ? (row[0] / row[1]) : 0.0;

}

inline bool reuse_flag_at(const std::vector<int>& all_reuse_flags, int n, int rank, int vi) {

	return all_reuse_flags[(size_t)rank * (size_t)n + (size_t)vi] != 0;

}

class Resizer {

	std::vector<std::pair<long,long>> remaining_;
	std::vector<long> remaining_offsets_;
	std::vector<long> target_cuts_;
	long total_rem_{0};

	struct VecTask {

		MalVec* v{nullptr};
		StagedBuffer gathered;

	};

	struct VecMeta {

		size_t esz{0};
		int shared_active{0};
		int ragged{0};

	};

	std::vector<VecTask> vtasks_;
	std::vector<VecMeta> vmeta_;
	std::vector<long> rem_per_rank_;
	std::vector<std::vector<char>> new_epoch_bufs_;
	std::vector<std::pair<long,long>> scratch_assigned_;
	std::vector<int> scratch_reuse_flags_;
	std::vector<int> scratch_all_reuse_flags_;

	std::vector<long> flat_buf_;
	std::vector<int> flat_counts_buf_;
	std::vector<int> flat_displs_buf_;
	std::vector<double> all_lb_buf_;
	std::vector<double> fused_gather_buf_;

	std::vector<TransferPlanEntry> build_transfer_plan(const std::vector<long>& old_vs) const;
	void init_vec_tasks(int n, int nvecs, bool was_active);
	void reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags) ;
	void exchange_vec_data(int n, bool was_active, const std::vector<long>& old_vs, const std::vector<TransferPlanEntry>& plan, const std::vector<int>& all_reuse_flags);

	void collect_ranges();
	std::vector<long> compute_target_cuts(long total_rem, int target);
	void redistribute_vecs(int n_vecs);
	void reduce_accs(int n_accs);
	void apply_active();
	void apply_inactive();
	void broadcast_shared_vecs();
	void broadcast_shared_mats();
	void stash_gather_cache();

	int target_;
	int old_a_size_{0};
	long my_new_vs_{0};
	long my_new_count_{0};
	long confirmed_snapshot_{LONG_MIN};
	std::vector<long> done_snapshot_;

public:

	explicit Resizer(int target) : target_(target) {}

	~Resizer() {

		for (auto& t : vtasks_) {

			if (t.gathered.ptr) {

				g_buffer_pool.release(t.gathered.ptr, t.gathered.bytes);

			}

		}

	}

	void prepare_phase();
	void commit_phase();

};

EpochMetrics gather_epoch_metrics() {

	EpochMetrics m;
	const int U = g.comm.u_size;

	double local_rem = 0.0;

	if (g.loop && g.comm.active != MPI_COMM_NULL) {

		const long cur = *g.loop->user_iter;

		if (cur + 1 < g.loop->end) {

			local_rem += (double)(g.loop->end - (cur + 1));

		}

		for (size_t ri = g.loop->plan_idx + 1; ri < g.loop->plan_ranges.size(); ri++) {

			local_rem += (double)(g.loop->plan_ranges[ri].second - g.loop->plan_ranges[ri].first);

		}

	}

	const double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	const long my_done = std::max(0L, g.lb.epoch_assigned - (long)local_rem);
	const double my_thr = (my_elapsed > kEpsElapsed && my_done > 0) ? (double)my_done / my_elapsed : 0.0;
	const double my_active_n = (double)g.comm.a_size;

	const double my_rem_time = (my_thr > kEpsThroughput) ? local_rem / my_thr : 0.0;

	const double my_has_loop = (g.loop && g.loop->phase.load(std::memory_order_relaxed) != MAL_LOOP_WAITING_ACTIVATION) ? 1.0 : 0.0;

	(void)U;

	const long iter_horizon = g.sync.iter_horizon.load(std::memory_order_acquire);
	double local_rem_total = local_rem;

	if (iter_horizon > 1 && g.loop && g.comm.active != MPI_COMM_NULL) {

		const long step_slice = g.loop->end - g.loop->start;
		local_rem_total += (double)step_slice * (double)(iter_horizon - 1);

	}

	double sum_in[3] = { local_rem_total, my_thr, my_rem_time };
	double sum_out[3] = { 0.0, 0.0, 0.0 };
	MPI_Allreduce(sum_in, sum_out, 3, MPI_DOUBLE, MPI_SUM, g.comm.universe);

	const double gate_elapsed_in = (g.comm.active != MPI_COMM_NULL) ? my_elapsed : -1.0;
	const double neg_thr_in = (g.comm.active != MPI_COMM_NULL && my_thr > kEpsThroughput) ? -my_thr : -1e300;
	const double settled_in = g.lb.last_decision_settled ? 1.0 : 0.0;
	double max_in[8] = { my_active_n, my_has_loop, my_thr, gate_elapsed_in, neg_thr_in, my_rem_time, (double)g.lb.my_slow_streak, settled_in };
	double max_out[8] = { 0.0, 0.0, 0.0, -1.0, -1e300, 0.0, 0.0, 0.0 };
	MPI_Allreduce(max_in, max_out, 8, MPI_DOUBLE, MPI_MAX, g.comm.universe);

	m.global_remaining = sum_out[0];
	m.global_thr = sum_out[1];
	m.sum_rem_time = sum_out[2];
	m.active_n = (int)std::lround(max_out[0]);
	m.any_has_loop = (max_out[1] > 0.5);
	m.max_thr = max_out[2];
	m.min_thr = (max_out[4] <= -1e299) ? 0.0 : -max_out[4];
	m.max_rem_time = max_out[5];
	m.max_slow_streak = (int)std::lround(max_out[6]);
	m.any_settled = (max_out[7] > 0.5);
	const double r0_elapsed = max_out[3];

	if (m.any_settled && g.sync.eval_stride.load(std::memory_order_relaxed) <= 0) {

		g.sync.eval_stride.store(LLONG_MAX, std::memory_order_release);
		MAL_LOG_L(MAL_LOG_DEBUG, "COST", "converged -> eval_stride latched (steady state: no more per-epoch consensus)");

	}

	if (m.active_n > 1 && m.sum_rem_time > kEpsThroughput) {

		const double avg_rem_time = m.sum_rem_time / (double)m.active_n;
		g.lb.my_slow_streak = (my_rem_time > kImbHi * avg_rem_time) ? (g.lb.my_slow_streak + 1) : 0;

	} else {

		g.lb.my_slow_streak = 0;

	}

	m.resize_commit_count = g.timing.resize_count;

	m.resize_cooldown_remaining = g.lb.resize_cooldown;
	m.rebalance_cooldown_remaining = g.lb.same_size_rebalance_cooldown;
	if (g.lb.resize_cooldown > 0) g.lb.resize_cooldown--;
	if (g.lb.same_size_rebalance_cooldown > 0) g.lb.same_size_rebalance_cooldown--;
	m.epoch_elapsed = r0_elapsed;
	m.epoch_interval_ms = g.cfg.epoch_ms.load(std::memory_order_relaxed);
	m.iterative_kernel = g.sync.iterative_kernel.load(std::memory_order_acquire);

	return m;

}

void Resizer::collect_ranges() {

	std::vector<long> local;
	local.reserve(16);

	done_snapshot_.clear();

	{

		std::lock_guard lk(g.sync.plan_mu);

		confirmed_snapshot_ = (g.loop && g.comm.active != MPI_COMM_NULL) ? g.loop->confirmed_iter.load(std::memory_order_acquire) : LONG_MIN;

		if (g.loop && g.comm.active != MPI_COMM_NULL) {

			const long first_s = confirmed_snapshot_ + 1;
			const long first_e = g.loop->end;

			if (first_s < first_e) {

				local.push_back(first_s);
				local.push_back(first_e);

			}

			for (size_t ri = g.loop->plan_idx + 1; ri < g.loop->plan_ranges.size(); ri++) {

				const long s = g.loop->plan_ranges[ri].first;
				const long e = g.loop->plan_ranges[ri].second;

				if (s < e) {

					local.push_back(s);
					local.push_back(e);

				}

			}

			done_snapshot_.reserve(g.loop->vecs.size());

			for (MalVec* v : g.loop->vecs) {

				if (v && !v->ragged && v->attach_policy == MAL_ATTACH_PARTITIONED) {

					done_snapshot_.push_back(std::clamp(confirmed_snapshot_ + 1 - v->buf_global_start, 0L, v->local_n));

				} else {

					done_snapshot_.push_back(-1);

				}

			}

			for (MalAcc* a : g.loop->accs) {

				if (a && !a->sealed && a->shadow_iter != LONG_MIN && a->shadow.size() >= a->esz) {

					a->capture.assign(a->shadow.begin(), a->shadow.begin() + (long)a->esz);
					a->capture_valid = true;

				} else if (a) {

					a->capture_valid = false;

				}

			}

		}

	}

	int my_count = (int)local.size();

	long my_rem_local = 0;

	for (size_t i = 0; i + 1 < local.size(); i += 2) {

		my_rem_local += local[i + 1] - local[i];

	}

	double my_elapsed = MPI_Wtime() - g.lb.epoch_start_time;
	long my_done = std::max(0L, g.lb.epoch_assigned - my_rem_local);

	double fused_send[kFusedGatherFields] = {(double)my_count, (double)my_done, my_elapsed};
	fused_gather_buf_.resize((size_t)g.comm.u_size * (size_t)kFusedGatherFields);

	MPI_Allgather(fused_send, kFusedGatherFields, MPI_DOUBLE, fused_gather_buf_.data(), kFusedGatherFields, MPI_DOUBLE, g.comm.universe);

	flat_counts_buf_.resize(g.comm.u_size);
	all_lb_buf_.resize((size_t)g.comm.u_size * (size_t)kLbGatherFields);

	for (int k = 0; k < g.comm.u_size; k++) {

		const double* row = &fused_gather_buf_[(size_t)k * (size_t)kFusedGatherFields];
		flat_counts_buf_[k] = (int)row[0];
		std::memcpy(&all_lb_buf_[(size_t)k * (size_t)kLbGatherFields], row + 1, kLbGatherFields * sizeof(double));

	}

	flat_displs_buf_ = make_displs(flat_counts_buf_);

	long total64 = (long)flat_displs_buf_.back() + (long)flat_counts_buf_.back();

	if (total64 > (long)INT_MAX) {

		MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "collect_ranges gather length %ld exceeds INT_MAX — too many fragmented ranges", total64);
		MPI_Abort(g.comm.universe, 1);

	}

	int total = (int)total64;

	if (total == 0) {

		remaining_.clear();
		remaining_offsets_.assign(1, 0);
		total_rem_ = 0;
		rem_per_rank_.assign(g.comm.u_size, 0);

		return;

	}

	flat_buf_.resize(total > 0 ? (size_t)total : 1);

	MPI_Allgatherv(local.empty() ? nullptr : local.data(), my_count, MPI_LONG, flat_buf_.data(), flat_counts_buf_.data(), flat_displs_buf_.data(), MPI_LONG, g.comm.universe);

	remaining_.clear();
	remaining_.reserve(total / 2 + 1);
	remaining_offsets_.clear();
	remaining_offsets_.reserve(total / 2 + 2);
	remaining_offsets_.push_back(0);

	total_rem_ = 0;
	rem_per_rank_.assign(g.comm.u_size, 0);

	for (int k = 0; k < g.comm.u_size; k++) {

		int disp = flat_displs_buf_[k];
		int nranges = flat_counts_buf_[k] / 2;

		for (int p = 0; p < nranges; p++) {

			long s = flat_buf_[disp + p * 2];
			long e = flat_buf_[disp + p * 2 + 1];
			long len = e - s;

			remaining_.push_back({s, e});
			remaining_offsets_.push_back(total_rem_ + len);
			rem_per_rank_[k] += len;
			total_rem_ += len;

		}

	}

	{

		std::lock_guard<std::mutex> lk(g.lb.weights_mu);

		if ((int)g.lb.weights.size() < g.comm.u_size) {

			g.lb.weights.assign((size_t)g.comm.u_size, 0.0);

		}

		const bool is_scale_up = (target_ > old_a_size_);

		if (is_scale_up) {

			const double w = 1.0 / (double)std::max(1, target_);

			for (int k = 0; k < g.comm.u_size; k++) {

				g.lb.weights[(size_t)k] = (k < target_) ? w : 0.0;

			}

		} else {

			double total_tp = 0.0;

			for (int k = 0; k < target_; k++) {

				total_tp += lb_row_tp(lb_row_at(all_lb_buf_, k));

			}

			if (total_tp > kEpsWeight) {

				for (int k = 0; k < g.comm.u_size; k++) {

					g.lb.weights[(size_t)k] = (k < target_) ? lb_row_tp(lb_row_at(all_lb_buf_, k)) / total_tp : 0.0;

				}

			} else {

				const double w = 1.0 / (double)std::max(1, target_);

				for (int k = 0; k < g.comm.u_size; k++) {

					g.lb.weights[(size_t)k] = (k < target_) ? w : 0.0;

				}

			}

		}

	}

	const double* my_row = lb_row_at(all_lb_buf_, g.comm.u_rank);
	MAL_LOG(MAL_LOG_DEBUG, "LB: epoch done=%.0f elapsed=%.3fs thr=%.1f iters/s weight=%.4f", my_row[0], my_row[1], lb_row_tp(my_row), g.comm.u_rank < (int)g.lb.weights.size() ? g.lb.weights[(size_t)g.comm.u_rank] : 0.0);

}

std::vector<long> Resizer::compute_target_cuts(long total_rem, int target) {

	const bool lb_enabled = g.cfg.load_balancing_enabled.load(std::memory_order_relaxed);

	if (!lb_enabled || target <= 0 || total_rem <= 0 || (int)all_lb_buf_.size() < g.comm.u_size * kLbGatherFields || (int)rem_per_rank_.size() < g.comm.u_size) {

		return build_partition_cuts(total_rem, target);

	}

	double sum_done = 0.0, sum_elapsed = 0.0;

	for (int k = 0; k < g.comm.u_size; k++) {

		const double* row = lb_row_at(all_lb_buf_, k);
		sum_done += row[0];
		sum_elapsed += row[1];

	}

	const double fallback_density = (sum_done > kEpsWeight && sum_elapsed > 0.0) ? sum_elapsed / sum_done : 1.0;

	std::vector<double> density((size_t)g.comm.u_size, fallback_density);
	double total_cost = 0.0;

	for (int k = 0; k < g.comm.u_size; k++) {

		const double* row = lb_row_at(all_lb_buf_, k);

		if (row[0] > kEpsWeight && row[1] > 0.0) {

			density[(size_t)k] = row[1] / row[0];

		}

		total_cost += (double)rem_per_rank_[k] * density[(size_t)k];

	}

	if (total_cost <= 0.0) {

		return build_partition_cuts(total_rem, target);

	}

	std::vector<long> cuts((size_t)target + 1, 0);
	cuts[(size_t)target] = total_rem;

	int seg = 0;
	double seg_cost_base = 0.0;
	long seg_row_base = 0;

	for (int r = 1; r < target; r++) {

		const double threshold = (double)r * total_cost / (double)target;

		while (seg < g.comm.u_size - 1) {

			const double seg_cost = (double)rem_per_rank_[seg] * density[(size_t)seg];

			if (seg_cost_base + seg_cost >= threshold) {

				break;

			}

			seg_cost_base += seg_cost;
			seg_row_base += rem_per_rank_[seg];
			seg++;

		}

		long cut_row = seg_row_base;
		const double d = density[(size_t)seg];

		if (d > 0.0) {

			cut_row = seg_row_base + (long)std::llround((threshold - seg_cost_base) / d);

		}

		if (cut_row < cuts[(size_t)r - 1]) {

			cut_row = cuts[(size_t)r - 1];

		}

		if (cut_row > total_rem) {

			cut_row = total_rem;

		}

		cuts[(size_t)r] = cut_row;

	}

	return cuts;

}

std::vector<TransferPlanEntry> Resizer::build_transfer_plan(const std::vector<long>& old_vs) const {

	std::vector<TransferPlanEntry> plan;
	plan.reserve((size_t)g.comm.u_size + (size_t)target_);

	if (target_ <= 0) {

		return plan;

	}

	int oi = 0;
	int ni = 0;
	long nv_s = target_cuts_[0];
	long nv_e = target_cuts_[1];

	while (oi < g.comm.u_size && rem_per_rank_[oi] == 0) {

		oi++;

	}

	while (oi < g.comm.u_size && ni < target_) {

		long ov_e = old_vs[(size_t)oi + 1];
		long seg_s = std::max(old_vs[(size_t)oi], nv_s);
		long seg_e = std::min(ov_e, nv_e);

		if (seg_s < seg_e) {

			plan.push_back({oi, ni, seg_s, seg_e - seg_s});

		}

		if (ov_e <= nv_e) {

			oi++;

			while (oi < g.comm.u_size && rem_per_rank_[oi] == 0) {

				oi++;

			}

		}

		if (nv_e <= ov_e) {

			ni++;

			if (ni < target_) {

				nv_s = target_cuts_[(size_t)ni];
				nv_e = target_cuts_[(size_t)ni + 1];

			}

		}

	}

	return plan;

}

void Resizer::init_vec_tasks(int n, int nvecs, bool was_active) {

	vtasks_.resize(n);

	if (g.gather_cache.size() < (size_t)n) {

		g.gather_cache.resize((size_t)n);

	}

	for (int vi = 0; vi < n; vi++) {

		auto& t = vtasks_[vi];

		t.v = (vi < nvecs) ? g.loop->vecs[vi] : nullptr;
		t.gathered = g.gather_cache[(size_t)vi];
		g.gather_cache[(size_t)vi] = {};

		if (vmeta_[vi].shared_active || vmeta_[vi].ragged) {

			if (t.v) {

				t.v->done_n = 0;

			}

			continue;

		}

		if (!t.v) {

			continue;

		}

		long old_done = t.v->done_n;
		long new_done = old_done;

		if (was_active && vi < (int)done_snapshot_.size() && done_snapshot_[(size_t)vi] >= 0) {

			new_done = done_snapshot_[(size_t)vi];

		}

		if (new_done > old_done) {

			append_done_segments(*t.v, *g.loop, t.v->plan_origin_n, old_done, new_done);

		}

		t.v->done_n = new_done;
		advance_read_only_cache_after_progress(*t.v, old_done, new_done);

	}

}

void Resizer::reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags) {

	if (!am_receiver || my_new_count <= 0) {

		return;

	}

	bool has_local_assignment = false;

	for (const auto& tr : plan) {

		if (tr.new_rank != g.comm.u_rank) {

			continue;

		}

		has_local_assignment = true;
		break;

	}

	if (!has_local_assignment) {

		return;

	}

	for (int vi = 0; vi < n; vi++) {

		if (vmeta_[vi].shared_active || vmeta_[vi].ragged || reuse_flag_at(all_reuse_flags, n, g.comm.u_rank, vi)) {

			continue;

		}

		size_t bytes = (size_t)my_new_count * vmeta_[vi].esz;
		void* gp = vtasks_[vi].gathered.ptr;
		size_t gc = vtasks_[vi].gathered.bytes;

		pool_reserve(gp, gc, bytes, false);
		vtasks_[vi].gathered = {gp, gc};

	}

}

void Resizer::exchange_vec_data(int n, bool was_active, const std::vector<long>& old_vs, const std::vector<TransferPlanEntry>& plan, const std::vector<int>& all_reuse_flags) {

	std::vector<MPI_Request> reqs;
	reqs.reserve(plan.size() * 2);

	std::vector<StagedBuffer> packed_sends;
	std::vector<StagedBuffer> packed_recvs;
	packed_sends.reserve(plan.size());
	packed_recvs.reserve(plan.size());

	struct PendingPackedRecv {

		const TransferPlanEntry* tr{nullptr};
		void* buf{nullptr};
		size_t bytes{0};

	};

	std::vector<PendingPackedRecv> pending_packed_recvs;
	pending_packed_recvs.reserve(plan.size());

	const int packed_tag = n;

	for (const auto& tr : plan) {

		const bool local_sender = (tr.old_rank == g.comm.u_rank);
		const bool local_recv = (tr.new_rank == g.comm.u_rank);

		if (!local_sender && !local_recv) {

			continue;

		}

		if (tr.old_rank == tr.new_rank) {

			if (!local_sender) {

				continue;

			}

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || vmeta_[vi].ragged || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				long bytes64 = tr.v_count * (long)esz;

				if (MAL_UNLIKELY(bytes64 > INT_MAX)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Transfer size overflow (%ld bytes) in redistribute_vecs", bytes64);
					MPI_Abort(g.comm.universe, 1);

				}

				int byte_count = (int)bytes64;

				if (byte_count == 0) {

					continue;

				}

				const char* send_base = (t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * esz : nullptr;

				if (send_base && t.gathered.ptr) {

					long src_off = (tr.v_start - old_vs[(size_t)tr.old_rank]) * (long)esz;
					long dst_off = (tr.v_start - my_new_vs_) * (long)esz;

					std::memmove(static_cast<char*>(t.gathered.ptr) + dst_off, send_base + src_off, byte_count);

				}

			}

			continue;

		}

		size_t packed_bytes = 0;
		int eligible_vecs = 0;

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || vmeta_[vi].ragged || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

				continue;

			}

			const size_t b = (size_t)tr.v_count * vmeta_[vi].esz;

			if (b > 0) {

				packed_bytes += b;
				eligible_vecs++;

			}

		}

		if (packed_bytes == 0) {

			continue;

		}

		if (packed_bytes > (size_t)INT_MAX || eligible_vecs == 1) {

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || vmeta_[vi].ragged || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				long bytes64 = tr.v_count * (long)esz;

				if (MAL_UNLIKELY(bytes64 > INT_MAX)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Transfer size overflow (%ld bytes) in redistribute_vecs", bytes64);
					MPI_Abort(g.comm.universe, 1);

				}

				int byte_count = (int)bytes64;

				if (byte_count == 0) {

					continue;

				}

				const char* send_base = ((tr.old_rank == g.comm.u_rank) && t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * esz : nullptr;

				if (tr.old_rank == g.comm.u_rank) {

					if (MAL_UNLIKELY(!send_base)) {

						MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing sender buffer for old_rank=%d vec=%d", tr.old_rank, vi);
						MPI_Abort(g.comm.universe, 1);

					}

					MPI_Request req;

					MPI_Isend(send_base + (tr.v_start - old_vs[(size_t)tr.old_rank]) * (long)esz, byte_count, MPI_BYTE, tr.new_rank, vi, g.comm.universe, &req);
					reqs.push_back(req);

				}

				if (tr.new_rank == g.comm.u_rank) {

					if (MAL_UNLIKELY(!t.gathered.ptr)) {

						MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing receiver buffer for new_rank=%d vec=%d", tr.new_rank, vi);
						MPI_Abort(g.comm.universe, 1);

					}

					char* dst = static_cast<char*>(t.gathered.ptr) + (tr.v_start - my_new_vs_) * (long)esz;
					MPI_Request req;

					MPI_Irecv(dst, byte_count, MPI_BYTE, tr.old_rank, vi, g.comm.universe, &req);
					reqs.push_back(req);

				}

			}

			continue;

		}

		if (local_sender) {

			void* send_buf = g_buffer_pool.acquire(packed_bytes);
			size_t off = 0;
			char* dst = static_cast<char*>(send_buf);

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || vmeta_[vi].ragged || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				const size_t bytes = (size_t)tr.v_count * esz;

				if (bytes == 0) {

					continue;

				}

				const char* send_base = (t.v && was_active) ? static_cast<char*>(t.v->buf) + t.v->done_n * esz : nullptr;

				if (MAL_UNLIKELY(!send_base)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing sender buffer while packing old_rank=%d vec=%d", tr.old_rank, vi);
					MPI_Abort(g.comm.universe, 1);

				}

				long src_off = (tr.v_start - old_vs[(size_t)tr.old_rank]) * (long)esz;
				std::memcpy(dst + off, send_base + src_off, bytes);
				off += bytes;

			}

			packed_sends.push_back({send_buf, packed_bytes});

			MPI_Request req;

			MPI_Isend(send_buf, (int)packed_bytes, MPI_BYTE, tr.new_rank, packed_tag, g.comm.universe, &req);
			reqs.push_back(req);

		}

		if (local_recv) {

			void* recv_buf = g_buffer_pool.acquire(packed_bytes);
			packed_recvs.push_back({recv_buf, packed_bytes});

			pending_packed_recvs.push_back({&tr, recv_buf, packed_bytes});

			MPI_Request req;

			MPI_Irecv(recv_buf, (int)packed_bytes, MPI_BYTE, tr.old_rank, packed_tag, g.comm.universe, &req);
			reqs.push_back(req);

		}

	}

	if (!reqs.empty()) {

		MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

	}

	for (const auto& pr : pending_packed_recvs) {

		if (pr.tr && pr.buf && pr.bytes > 0) {

			size_t off = 0;
			const TransferPlanEntry& tr = *pr.tr;
			const char* src = static_cast<const char*>(pr.buf);

			for (int vi = 0; vi < n; vi++) {

				if (vmeta_[vi].shared_active || vmeta_[vi].ragged || reuse_flag_at(all_reuse_flags, n, tr.new_rank, vi)) {

					continue;

				}

				auto& t = vtasks_[vi];
				const size_t esz = vmeta_[vi].esz;
				const size_t bytes = (size_t)tr.v_count * esz;

				if (bytes == 0) {

					continue;

				}

				if (MAL_UNLIKELY(!t.gathered.ptr)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing receiver buffer while unpacking new_rank=%d vec=%d", tr.new_rank, vi);
					MPI_Abort(g.comm.universe, 1);

				}

				long dst_off = (tr.v_start - my_new_vs_) * (long)esz;
				std::memcpy(static_cast<char*>(t.gathered.ptr) + dst_off, src + off, bytes);
				off += bytes;

			}

		}

	}

	for (auto& b : packed_sends) {

		g_buffer_pool.release(b.ptr, b.bytes);

	}

	for (auto& b : packed_recvs) {

		g_buffer_pool.release(b.ptr, b.bytes);

	}

}

void Resizer::redistribute_vecs(int n) {

	if (n == 0) {

		return;

	}

	int nvecs = g.loop ? (int)g.loop->vecs.size() : 0;

	std::vector<long> old_vs(g.comm.u_size + 1, 0);
	std::inclusive_scan(rem_per_rank_.begin(), rem_per_rank_.end(), old_vs.begin() + 1);
	target_cuts_.clear();

	if (target_ > 0) {

		target_cuts_ = compute_target_cuts(total_rem_, target_);

	}

	const auto plan = build_transfer_plan(old_vs);

	bool was_active = (g.comm.active != MPI_COMM_NULL);
	init_vec_tasks(n, nvecs, was_active);

	my_new_vs_ = 0;
	my_new_count_ = 0;
	bool am_receiver = (g.comm.u_rank < target_);

	if (am_receiver) {

		long my_nv_e = target_cuts_[(size_t)g.comm.u_rank + 1];
		my_new_vs_ = target_cuts_[(size_t)g.comm.u_rank];
		my_new_count_ = my_nv_e - my_new_vs_;

	}

	scratch_assigned_.clear();

	if (am_receiver && my_new_count_ > 0) {

		scratch_assigned_ = slice_remaining(remaining_, remaining_offsets_, my_new_vs_, my_new_vs_ + my_new_count_);

	}

	scratch_reuse_flags_.assign((size_t)n, 0);

	if (am_receiver && my_new_count_ > 0) {

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || vmeta_[vi].ragged || !vtasks_[vi].v) {

				continue;

			}

			if (vec_can_reuse_assigned_ranges(*vtasks_[vi].v, scratch_assigned_)) {

				scratch_reuse_flags_[(size_t)vi] = 1;

			}

		}

	}

	scratch_all_reuse_flags_.assign((size_t)g.comm.u_size * (size_t)n, 0);

	MPI_Allgather(scratch_reuse_flags_.data(), n, MPI_INT, scratch_all_reuse_flags_.data(), n, MPI_INT, g.comm.universe);

	reserve_receiver_buffers(n, am_receiver, plan, my_new_count_, scratch_all_reuse_flags_);

	int n_sources = 0;
	int source_rank = -1;

	for (int k = 0; k < g.comm.u_size; k++) {

		if (rem_per_rank_[k] > 0) {

			n_sources++;
			source_rank = k;

		}

	}

	if (n_sources == 1 && target_ > 1) {

		std::vector<int> scounts(g.comm.u_size, 0);
		std::vector<int> sdispls(g.comm.u_size, 0);

		for (int vi = 0; vi < n; vi++) {

			if (vmeta_[vi].shared_active || vmeta_[vi].ragged) {

				continue;

			}

			const size_t esz = vmeta_[vi].esz;
			auto& t = vtasks_[vi];

			const char* sendbuf = nullptr;

			if (g.comm.u_rank == source_rank && t.v && was_active) {

				sendbuf = static_cast<char*>(t.v->buf) + t.v->done_n * (long)esz;

				for (int j = 0; j < target_; j++) {

					if (reuse_flag_at(scratch_all_reuse_flags_, n, j, vi)) {

						scounts[j] = 0;
						sdispls[j] = 0;

					} else {

						long count = target_cuts_[(size_t)j + 1] - target_cuts_[(size_t)j];
						long bytes = count * (long)esz;

						scounts[j] = (bytes <= INT_MAX) ? (int)bytes : 0;
						sdispls[j] = (int)(target_cuts_[(size_t)j] * (long)esz);

					}

				}

				for (int j = target_; j < g.comm.u_size; j++) {

					scounts[j] = 0;
					sdispls[j] = 0;

				}

			}

			const bool my_reuse = reuse_flag_at(scratch_all_reuse_flags_, n, g.comm.u_rank, vi);
			int recvcount = (am_receiver && !my_reuse && my_new_count_ > 0 && t.gathered.ptr) ? (int)(my_new_count_ * (long)esz) : 0;

			MPI_Scatterv(sendbuf, scounts.data(), sdispls.data(), MPI_BYTE, t.gathered.ptr, recvcount, MPI_BYTE, source_rank, g.comm.universe);

		}

	} else {

		exchange_vec_data(n, was_active, old_vs, plan, scratch_all_reuse_flags_);

	}

}

void Resizer::reduce_accs(int n) {

	if (n == 0) {

		return;

	}

	int naccs = g.loop ? (int)g.loop->accs.size() : 0;

	new_epoch_bufs_.resize(n);
	std::vector<MalAcc*>& loop_accs = g.loop->accs;

	struct AccGetter {

		int naccs;
		const std::vector<MalAcc*>* accs;

		MalAcc* operator()(int k) const {

			return k < naccs ? (*accs)[(size_t)k] : nullptr;

		}

	};

	struct AccResultSetter {

		int naccs;
		const std::vector<MalAcc*>* accs;
		std::vector<std::vector<char>>* epoch_bufs;

		void operator()(int k, const char* r, int esz) const {

			(*epoch_bufs)[(size_t)k].assign(r, r + esz);

			if (k >= naccs) {

				return;

			}

			MalAcc* a = (*accs)[(size_t)k];

			if (a->sealed) {

				return;

			}

			a->epoch_buf.assign(r, r + esz);
			a->needs_reset = true;

		}

	};

	batched_allreduce(n, AccGetter{naccs, &loop_accs}, AccResultSetter{naccs, &loop_accs, &new_epoch_bufs_});

}

void resync_ragged_vecs(const std::vector<std::pair<long,long>>& my_ranges) {

	if (g.comm.active == MPI_COMM_NULL || g.loop == nullptr) {

		return;

	}

	bool any_ragged = false;

	for (MalVec* v : g.loop->vecs) {

		if (v && v->ragged) { any_ragged = true; break; }

	}

	if (!any_ragged) {

		return;

	}

	const int asz = g.comm.a_size;
	const int arank = g.comm.a_rank;

	std::vector<long> myflat;
	myflat.reserve(my_ranges.size() * 2);

	for (auto [s, e] : my_ranges) {

		myflat.push_back(s);
		myflat.push_back(e);

	}

	int mycount = (int)myflat.size();
	std::vector<int> counts((size_t)asz, 0);
	MPI_Allgather(&mycount, 1, MPI_INT, counts.data(), 1, MPI_INT, g.comm.active);

	std::vector<int> displs = make_displs(counts);
	long total = (long)displs.back() + counts.back();
	std::vector<long> allflat((size_t)std::max(1L, total));
	MPI_Allgatherv(myflat.data(), mycount, MPI_LONG, allflat.data(), counts.data(), displs.data(), MPI_LONG, g.comm.active);

	for (MalVec* v : g.loop->vecs) {

		if (!v || !v->ragged) {

			continue;

		}

		const long* ro = v->ragged_row_offsets;
		const size_t esz = v->elem_size;

		std::vector<int> sc((size_t)asz, 0);
		long my_nnz = 0;

		for (int k = 0; k < asz; k++) {

			long nnz_k = 0;

			for (int p = 0; p < counts[(size_t)k]; p += 2) {

				const long s = allflat[(size_t)displs[(size_t)k] + p];
				const long e = allflat[(size_t)displs[(size_t)k] + p + 1];
				nnz_k += ro[e] - ro[s];

			}

			const long bytes = nnz_k * (long)esz;

			if (MAL_UNLIKELY(bytes > INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "ragged resync per-rank size overflow (%ld bytes)", bytes);
				MPI_Abort(g.comm.universe, 1);

			}

			sc[(size_t)k] = (int)bytes;

			if (k == arank) {

				my_nnz = nnz_k;

			}

		}

		std::vector<int> sd = make_displs(sc);

		std::vector<char> sendbuf;

		if (arank == 0 && v->ragged_full_src) {

			const long send_total = (long)sd.back() + sc.back();
			sendbuf.resize((size_t)std::max(1L, send_total));
			long off = 0;

			for (int k = 0; k < asz; k++) {

				for (int p = 0; p < counts[(size_t)k]; p += 2) {

					const long s = allflat[(size_t)displs[(size_t)k] + p];
					const long e = allflat[(size_t)displs[(size_t)k] + p + 1];
					const long nb = (ro[e] - ro[s]) * (long)esz;

					if (nb > 0) {

						std::memcpy(sendbuf.data() + off, static_cast<char*>(v->ragged_full_src) + ro[s] * (long)esz, (size_t)nb);
						off += nb;

					}

				}

			}

		}

		const long recv_bytes = my_nnz * (long)esz;

		g_buffer_pool.release(v->buf, v->buf_bytes > 0 ? v->buf_bytes : 1);
		v->buf = static_cast<char*>(g_buffer_pool.acquire((size_t)std::max(1L, recv_bytes)));
		v->buf_bytes = (size_t)std::max(1L, recv_bytes);
		v->local_n = my_nnz;
		v->done_n = 0;
		v->plan_origin_n = 0;

		MPI_Scatterv(arank == 0 ? sendbuf.data() : nullptr, arank == 0 ? sc.data() : nullptr, arank == 0 ? sd.data() : nullptr, MPI_BYTE, v->buf, (int)recv_bytes, MPI_BYTE, 0, g.comm.active);

		v->ragged_bases.clear();
		long cum = 0;

		for (auto [s, e] : my_ranges) {

			v->ragged_bases.push_back(cum);
			cum += ro[e] - ro[s];

		}

		if (v->ragged_bases.empty()) {

			v->ragged_bases.push_back(0);

		}

		if (!my_ranges.empty()) {

			v->buf_global_start = ro[my_ranges[0].first];
			v->sync_user_ptr();

		}

	}

}

static void apply_pending_acc_resets() {

	if (g.loop == nullptr) {

		return;

	}

	std::lock_guard lk(g.sync.plan_mu);

	for (MalAcc* a : g.loop->accs) {

		if (a && a->needs_reset) {

			write_identity(static_cast<char*>(a->ptr), a->dtype_idx, a->dop_idx, (int)a->esz);
			a->shadow_iter = LONG_MIN;
			a->capture_valid = false;
			a->needs_reset = false;

		}

	}

}

void Resizer::apply_active() {

	apply_pending_acc_resets();

	if (target_cuts_.size() != (size_t)g.comm.a_size + 1) {

		target_cuts_ = compute_target_cuts(total_rem_, g.comm.a_size);

	}

	const std::vector<long>& active_cuts = target_cuts_;
	long vstart = active_cuts[(size_t)g.comm.a_rank];
	long vend = active_cuts[(size_t)g.comm.a_rank + 1];
	scratch_assigned_ = slice_remaining(remaining_, remaining_offsets_, vstart, vend);
	auto& assigned = scratch_assigned_;

	long new_asgn = vend - vstart;
	const bool has_assigned_ranges = (new_asgn > 0 && !assigned.empty());

	const bool waiting_for_activation = g.loop && g.loop->phase.load(std::memory_order_acquire) == MAL_LOOP_WAITING_ACTIVATION;
	bool publish_pending_after_broadcast = false;
	std::vector<std::pair<long,long>> deferred_pending_ranges;

	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "a_rank=%d assigned %zu range(s) (%ld iters, weight=%.4f)", g.comm.a_rank, assigned.size(), new_asgn, g.comm.a_rank < (int)g.lb.weights.size() ? g.lb.weights[g.comm.a_rank] : 1.0 / g.comm.a_size);

	g.lb.epoch_assigned = new_asgn;
	g.lb.epoch_start_time = MPI_Wtime();

	if (g.loop && !waiting_for_activation) {

		for (size_t ti = 0; ti < vtasks_.size(); ti++) {

			auto& t = vtasks_[ti];

			if (!t.v) {

				continue;

			}

			if (t.v->attach_policy == MAL_ATTACH_SHARED_ACTIVE || t.v->attach_policy == MAL_ATTACH_SHARED_ALL) {

				configure_shared_active_vec(*t.v, (size_t)std::max(1L, t.v->total_N) * vmeta_[ti].esz);

				continue;

			}

			if (t.v->ragged) {

				continue;

			}

			long new_local = t.v->done_n + new_asgn;
			bool reused_local = false;

			if (has_assigned_ranges && !t.gathered.ptr) {

				reused_local = vec_reuse_local_copy(*t.v, assigned, t.v->done_n);

			}

			size_t buf_need = (size_t)std::max(1L, new_local) * vmeta_[ti].esz;
			pool_reserve(t.v->buf, t.v->buf_bytes, buf_need);

			if (has_assigned_ranges && !reused_local && t.gathered.ptr) {

				long buf_off = t.v->done_n;
				long gathered_off = 0;

				for (auto [g_start, g_end] : assigned) {

					long len = g_end - g_start;

					if (len > 0 && gathered_off + len <= my_new_count_) {

						std::memcpy(static_cast<char*>(t.v->buf) + buf_off * vmeta_[ti].esz, static_cast<char*>(t.gathered.ptr) + gathered_off * vmeta_[ti].esz, len * vmeta_[ti].esz);

					}

					buf_off += len;
					gathered_off += len;

				}

			}

			long new_buf_global_start = assigned.empty() ? t.v->buf_global_start : (assigned[0].first - t.v->done_n);
			set_partitioned_layout(*t.v, new_local, t.v->done_n, new_buf_global_start);

			if (!set_read_only_cache_from_ranges(*t.v, assigned, t.v->done_n) && t.v->access_mode != MAL_ACCESS_READ_ONLY) {

				t.v->cache_valid = false;

			}

			t.v->sync_user_ptr();

		}

		if (!assigned.empty()) {

			install_loop_plan(*g.loop, assigned);
			set_limit(*g.loop, g.loop->end);
			set_iter (*g.loop, g.loop->start - 1);

			g.loop->confirmed_iter.store(g.loop->start - 1, std::memory_order_release);
			g.sync.loop_has_new_work.store(true, std::memory_order_release);

		} else {

			freeze_loop_at_current(*g.loop);

		}

	} else {

		auto pa = std::make_unique<PendingActivation>();

		deferred_pending_ranges = std::move(assigned);
		pa->ranges.clear();
		pa->vec_slices.resize(vtasks_.size());

		for (int vi = 0; vi < (int)vtasks_.size(); vi++) {

			auto& t = vtasks_[vi];

			if (new_asgn > 0 && t.gathered.ptr && vmeta_[vi].esz > 0) {

				pa->vec_slices[(size_t)vi] = t.gathered;
				t.gathered = {};

			}

		}

		pa->acc_epoch_bufs = std::move(new_epoch_bufs_);
		g.pending = std::move(pa);
		publish_pending_after_broadcast = true;

	}

	resync_ragged_vecs(publish_pending_after_broadcast ? deferred_pending_ranges : assigned);

	if (target_ > old_a_size_) {

		broadcast_shared_vecs();
		broadcast_shared_mats();

	}

	if (publish_pending_after_broadcast && g.pending) {

		g.pending->ranges = std::move(deferred_pending_ranges);

		if (!g.pending->ranges.empty()) {

			g.sync.pending_has_ranges.store(true, std::memory_order_release);

		}

		g.sync.notify();

	}

}

void Resizer::broadcast_shared_mats() {

	if (MAL_UNLIKELY(g.comm.active == MPI_COMM_NULL || target_ <= old_a_size_)) {

		return;

	}

	int n_shared = (int)g.shared.size();

	MPI_Bcast(&n_shared, 1, MPI_INT, 0, g.comm.active);

	if (n_shared == 0) {

		return;

	}

	const bool is_new = (g.comm.u_rank >= old_a_size_ && g.comm.u_rank < target_);

	std::vector<size_t> tots(n_shared, 0);

	if (!is_new) {

		for (int si = 0; si < n_shared; si++) {

			tots[si] = get_shared_mat_or_abort(si)->total_bytes;

		}

	}

	mpi_bcast_bytes(tots.data(), (size_t)n_shared * sizeof(size_t), 0, g.comm.active);

	constexpr int kSharedMatTagBase = 0x2000;

	auto entry_needed = [](unsigned long long mask, int si) {

		return si >= 64 || ((mask >> si) & 1ull) != 0;

	};

	unsigned long long my_need = 0;

	if (is_new) {

		for (int si = 0; si < n_shared && si < 64; si++) {

			SharedMat* sm = (si < (int)g.shared.size()) ? g.shared[(size_t)si].get() : nullptr;

			if (!(sm && sm->buf && sm->total_bytes == tots[si])) {

				my_need |= (1ull << si);

			}

		}

	}

	std::vector<unsigned long long> all_need((size_t)g.comm.a_size, 0);
	MPI_Allgather(&my_need, 1, MPI_UNSIGNED_LONG_LONG, all_need.data(), 1, MPI_UNSIGNED_LONG_LONG, g.comm.active);

	if (is_new) {

		for (int si = 0; si < n_shared; si++) {

			const size_t tot = tots[si];
			const size_t cap = tot > 0 ? tot : 1;
			SharedMat* sm = (si < (int)g.shared.size()) ? g.shared[(size_t)si].get() : nullptr;

			if (sm && sm->buf && sm->total_bytes == tot) {

				if (tot > 0 && entry_needed(my_need, si)) {

					mpi_recv_bytes(sm->buf, tot, 0, kSharedMatTagBase + si, g.comm.active);

				}

				continue;

			}

			PendingActivation& pa = ensure_pending_activation();
			void* buf = g_buffer_pool.acquire(cap);

			if (tot > 0) {

				mpi_recv_bytes(buf, tot, 0, kSharedMatTagBase + si, g.comm.active);

			}

			pa.shared_mats.push_back({buf, cap});

		}

		return;

	}

	if (g.comm.a_rank != 0) {

		return;

	}

	for (int nr = old_a_size_; nr < target_; nr++) {

		for (int si = 0; si < n_shared; si++) {

			const size_t tot = tots[si];

			if (tot == 0 || !entry_needed(all_need[(size_t)nr], si)) {

				continue;

			}

			mpi_send_bytes(get_shared_mat_or_abort(si)->buf, tot, nr, kSharedMatTagBase + si, g.comm.active);

		}

	}

}

void Resizer::broadcast_shared_vecs() {

	if (MAL_UNLIKELY(g.comm.active == MPI_COMM_NULL || target_ <= old_a_size_)) {

		return;

	}

	struct SharedVecBroadcast {

		int index;
		int bytes;

	};

	std::vector<SharedVecBroadcast> shared_meta;

	if (g.comm.a_rank == 0) {

		shared_meta.reserve(g.vecs.size());

		for (int i = 0; i < (int)g.vecs.size(); i++) {

			MalVec* v = g.vecs[i].get();

			if (!v || (v->attach_policy != MAL_ATTACH_SHARED_ACTIVE && v->attach_policy != MAL_ATTACH_SHARED_ALL)) {

				continue;

			}

			size_t b = (size_t)std::max(0L, v->total_N) * v->elem_size;

			if (MAL_UNLIKELY(b > (size_t)INT_MAX)) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector size overflow (%zu bytes)", b);
				MPI_Abort(g.comm.universe, 1);

			}

			shared_meta.push_back({i, (int)b});

		}

	}

	int n = (int)shared_meta.size();
	MPI_Bcast(&n, 1, MPI_INT, 0, g.comm.active);

	if (n == 0) {

		return;

	}

	shared_meta.resize(n);
	MPI_Bcast(shared_meta.data(), n * (int)sizeof(SharedVecBroadcast), MPI_BYTE, 0, g.comm.active);

	const bool is_new = (g.comm.u_rank >= old_a_size_ && g.comm.u_rank < target_);
	constexpr int kSharedVecTagBase = 0x1000;

	auto entry_needed = [](unsigned long long mask, int mi) {

		return mi >= 64 || ((mask >> mi) & 1ull) != 0;

	};

	unsigned long long my_need = 0;

	if (is_new) {

		for (int mi = 0; mi < n; mi++) {

			if (mi >= 64) {

				break;

			}

			const int vi = shared_meta[(size_t)mi].index;
			const int nbytes = shared_meta[(size_t)mi].bytes;
			MalVec* v = (vi >= 0 && vi < (int)g.vecs.size()) ? g.vecs[(size_t)vi].get() : nullptr;

			const bool reusable = v && v->attach_policy == MAL_ATTACH_SHARED_ALL && v->access_mode == MAL_ACCESS_READ_ONLY && v->buf && v->local_n == v->total_N && (size_t)std::max(0, nbytes) <= (size_t)std::max(1L, v->total_N) * v->elem_size;

			if (!reusable) {

				my_need |= (1ull << mi);

			}

		}

	}

	std::vector<unsigned long long> all_need((size_t)g.comm.a_size, 0);
	MPI_Allgather(&my_need, 1, MPI_UNSIGNED_LONG_LONG, all_need.data(), 1, MPI_UNSIGNED_LONG_LONG, g.comm.active);

	if (is_new) {

		for (int mi = 0; mi < n; mi++) {

			const int vi = shared_meta[(size_t)mi].index;
			const int nbytes = shared_meta[(size_t)mi].bytes;

			MalVec* v = (vi >= 0 && vi < (int)g.vecs.size()) ? g.vecs[(size_t)vi].get() : nullptr;
			const bool have_vec = v && (v->attach_policy == MAL_ATTACH_SHARED_ACTIVE || v->attach_policy == MAL_ATTACH_SHARED_ALL);

			if (have_vec) {

				const size_t buf_need = (size_t)std::max(1L, v->total_N) * v->elem_size;

				if (MAL_UNLIKELY((size_t)std::max(0, nbytes) > buf_need)) {

					MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared vector size mismatch at index %d (%d > %zu)", vi, nbytes, buf_need);
					MPI_Abort(g.comm.universe, 1);

				}

				configure_shared_active_vec(*v, buf_need);

				if (nbytes > 0 && entry_needed(my_need, mi)) {

					mpi_recv_bytes(v->buf, (size_t)nbytes, 0, kSharedVecTagBase + mi, g.comm.active);

				}

				continue;

			}

			PendingActivation& pa = ensure_pending_activation();
			const size_t cap = nbytes > 0 ? (size_t)nbytes : 1;
			void* buf = g_buffer_pool.acquire(cap);

			if (nbytes > 0) {

				mpi_recv_bytes(buf, (size_t)nbytes, 0, kSharedVecTagBase + mi, g.comm.active);

			}

			pa.shared_vecs.push_back({buf, cap});

		}

		return;

	}

	if (g.comm.a_rank != 0) {

		return;

	}

	for (int nr = old_a_size_; nr < target_; nr++) {

		for (int mi = 0; mi < n; mi++) {

			const int vi = shared_meta[(size_t)mi].index;
			const int nbytes = shared_meta[(size_t)mi].bytes;

			if (nbytes <= 0 || !entry_needed(all_need[(size_t)nr], mi)) {

				continue;

			}

			if (MAL_UNLIKELY(vi < 0 || vi >= (int)g.vecs.size() || !g.vecs[(size_t)vi])) {

				MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Shared-active vector index out of range: %d", vi);
				MPI_Abort(g.comm.universe, 1);

			}

			mpi_send_bytes(g.vecs[(size_t)vi]->buf, (size_t)nbytes, nr, kSharedVecTagBase + mi, g.comm.active);

		}

	}

}

void Resizer::apply_inactive() {

	apply_pending_acc_resets();

	g.comm.a_rank = -1;
	g.comm.a_size = 0;

	g.lb.epoch_assigned = 0;
	g.lb.epoch_start_time = 0.0;

	for (auto& t : vtasks_) {

		if (!t.v) {

			continue;

		}

		if (t.v->attach_policy == MAL_ATTACH_SHARED_ACTIVE) {

			release_shared_active_vec(*t.v);

			continue;

		}

		if (t.v->ragged) {

			continue;

		}

		if (t.v->attach_policy == MAL_ATTACH_SHARED_ALL) {

			configure_shared_active_vec(*t.v, (size_t)std::max(1L, t.v->total_N) * t.v->elem_size);

			continue;

		}

		refresh_inactive_read_only_cache(*t.v);

		if (t.v->access_mode != MAL_ACCESS_READ_ONLY) {

			size_t buf_need = (size_t)std::max(1L, t.v->done_n) * t.v->elem_size;
			pool_reserve(t.v->buf, t.v->buf_bytes, buf_need);
			t.v->local_n = t.v->done_n;

		}

		t.v->sync_user_ptr();

	}

	if (g.loop) {

		freeze_loop_at_current(*g.loop);

	}

	g.sync.compute_ready.store(true, std::memory_order_release);

}

void Resizer::stash_gather_cache() {

	if (g.gather_cache.size() < vtasks_.size()) {

		g.gather_cache.resize(vtasks_.size());

	}

	for (size_t i = 0; i < vtasks_.size(); i++) {

		auto& t = vtasks_[i];
		auto& c = g.gather_cache[i];

		if (c.ptr) {

			g_buffer_pool.release(c.ptr, c.bytes);

		}

		c = t.gathered;
		t.gathered = {};

	}

}

void Resizer::prepare_phase() {

	const double t0 = MPI_Wtime();
	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Prepare phase start target=%d (active=%d)", target_, g.comm.a_size);

	old_a_size_ = g.comm.a_size;
	MPI_Bcast(&old_a_size_, 1, MPI_INT, 0, g.comm.universe);

	collect_ranges();

	int nvecs = g.loop ? (int)g.loop->vecs.size() : 0;
 	int naccs = g.loop ? (int)g.loop->accs.size() : 0;

	int header[3] = {nvecs, naccs, 0};
	header[2] = (int)(3 * sizeof(int) + (size_t)nvecs * sizeof(VecMeta));
	int bcast_meta_bytes = header[2];
	std::vector<char> bcast_buf((size_t)bcast_meta_bytes);

	std::memcpy(bcast_buf.data(), header, 3 * sizeof(int));

	if (nvecs > 0) {

		vmeta_.resize(nvecs);

		for (int vi = 0; vi < nvecs; vi++) {

			vmeta_[vi].esz = g.loop->vecs[vi]->elem_size;
			vmeta_[vi].shared_active = (g.loop->vecs[vi]->attach_policy == MAL_ATTACH_SHARED_ACTIVE || g.loop->vecs[vi]->attach_policy == MAL_ATTACH_SHARED_ALL) ? 1 : 0;
			vmeta_[vi].ragged = g.loop->vecs[vi]->ragged ? 1 : 0;

		}

		std::memcpy(bcast_buf.data() + 3 * sizeof(int), vmeta_.data(), (size_t)nvecs * sizeof(VecMeta));

	}

	MPI_Bcast(header, 3, MPI_INT, 0, g.comm.universe);

	bcast_meta_bytes = header[2];
	bcast_buf.resize((size_t)bcast_meta_bytes);

	if (bcast_meta_bytes > (int)(3 * sizeof(int))) {

		MPI_Bcast(bcast_buf.data() + 3 * sizeof(int), bcast_meta_bytes - (int)(3 * sizeof(int)), MPI_BYTE, 0, g.comm.universe);

	}

	int n_vecs = header[0];
	int n_accs = header[1];

	if (n_vecs > 0) {

		vmeta_.resize(n_vecs);
		std::memcpy(vmeta_.data(), bcast_buf.data() + 3 * sizeof(int), (size_t)n_vecs * sizeof(VecMeta));

	}

	redistribute_vecs(n_vecs);
	reduce_accs(n_accs);

	if (g.comm.u_rank == 0 && !target_cuts_.empty() && total_rem_ > 0) {

		char buf[4096];
		int pos = 0;
		pos += snprintf(buf + pos, (int)sizeof(buf) - pos, "Distribution after resize %d->%d (total=%ld iters):", old_a_size_, target_, total_rem_);

		for (int k = 0; k < target_ && pos < (int)sizeof(buf) - 64; k++) {

			long iters = target_cuts_[(size_t)k + 1] - target_cuts_[(size_t)k];
			double w = k < (int)g.lb.weights.size() ? g.lb.weights[(size_t)k] : 1.0 / target_;
			pos += snprintf(buf + pos, (int)sizeof(buf) - pos, "\n R%-2d: %6ld iters weight=%.4f", k, iters, w);

		}

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "%s", buf);

	}

	const double prepare_elapsed = MPI_Wtime() - t0;
	g.timing.resize_prepare += prepare_elapsed;
	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Prepare phase done target=%d in %.4f s", target_, prepare_elapsed);

}

void Resizer::commit_phase() {

	{

		const double t_wfc = MPI_Wtime();
		g.sync.wait_for_compute();
		g.timing.wait_for_compute += MPI_Wtime() - t_wfc;

	}

	MPI_Barrier(g.comm.universe);

	if (g.sync.step_buf != nullptr && g.loop != nullptr && g.comm.active != MPI_COMM_NULL) {

		mal_allgather_replicated(*g.loop, g.sync.step_buf, g.sync.step_elem, g.sync.step_total_n);

	}

	MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Commit phase start target=%d (current=%d)", target_, g.comm.a_size);

	double t0 = MPI_Wtime();

	const bool same_size_rebalance = (target_ == old_a_size_);

	g.lb.last_commit_grew = (target_ > old_a_size_);

	if (!same_size_rebalance) {

		if (g.comm.active != MPI_COMM_NULL) {

			int rc = MPI_Comm_free(&g.comm.active);

			if (rc != MPI_SUCCESS) {

				char err[MPI_MAX_ERROR_STRING] = {};
				int len = 0;
				MPI_Error_string(rc, err, &len);
				MAL_LOG_L(MAL_LOG_ERROR, "MPI", "MPI_Comm_free(active, commit) failed rc=%d msg=%.*s", rc, len, err);

			}

			g.comm.active = MPI_COMM_NULL;

		}

		int color = (g.comm.u_rank < target_) ? 0 : MPI_UNDEFINED;
		int rc = MPI_Comm_split(g.comm.universe, color, g.comm.u_rank, &g.comm.active);

		if (rc != MPI_SUCCESS) {

			char err[MPI_MAX_ERROR_STRING] = {};
			int len = 0;
			MPI_Error_string(rc, err, &len);
			MAL_LOG_L(MAL_LOG_ERROR, "MPI", "MPI_Comm_split(active, commit) failed rc=%d msg=%.*s", rc, len, err);

		}

	}

	if (g.comm.active != MPI_COMM_NULL) {

		int rc = MPI_Comm_set_errhandler(g.comm.active, MPI_ERRORS_RETURN);

		if (rc != MPI_SUCCESS) {

			char err[MPI_MAX_ERROR_STRING] = {};
			int len = 0;
			MPI_Error_string(rc, err, &len);
			MAL_LOG_L(MAL_LOG_ERROR, "MPI", "MPI_Comm_set_errhandler(active, commit) failed rc=%d msg=%.*s", rc, len, err);

		}

		MPI_Comm_rank(g.comm.active, &g.comm.a_rank);
		MPI_Comm_size(g.comm.active, &g.comm.a_size);
		apply_active();

	} else {

		apply_inactive();

	}

	stash_gather_cache();

	const double commit_elapsed = MPI_Wtime() - t0;
	const double epoch_secs = std::max(kEpsElapsed, g.cfg.epoch_ms.load() / 1000.0);
	const bool fast_resp = g.cfg.fast_response.load(std::memory_order_relaxed);

	double max_commit_elapsed = commit_elapsed;
	MPI_Allreduce(MPI_IN_PLACE, &max_commit_elapsed, 1, MPI_DOUBLE, MPI_MAX, g.comm.universe);

	const int adaptive_cooldown = fast_resp ? 0 : std::max(0, (int)std::ceil(max_commit_elapsed / epoch_secs));
	const int min_cooldown = fast_resp ? 0 : kMinResizeCooldownEpochs;
	const int base_resize_cooldown = std::max(adaptive_cooldown, min_cooldown);

	if (same_size_rebalance) {

		g.lb.same_size_rebalance_cooldown = std::max(adaptive_cooldown, 1);

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Rebalance on %d active ranks done in %.4f s (cooldown=%d)", target_, commit_elapsed, g.lb.same_size_rebalance_cooldown);

	} else {

		const bool is_oscillation = (g.lb.prev_resize_to > 0 && target_ == g.lb.prev_resize_from && old_a_size_ == g.lb.prev_resize_to);

		g.lb.prev_resize_from = old_a_size_;
		g.lb.prev_resize_to = target_;

		g.lb.my_slow_streak = 0;

		g.lb.resize_cooldown = (is_oscillation && !fast_resp) ? std::max(base_resize_cooldown * 2, 4) : base_resize_cooldown;

		MAL_LOG_L(MAL_LOG_DEBUG, "RESIZE", "Resize %d->%d done in %.4f s (cooldown=%d%s)", old_a_size_, target_, commit_elapsed, g.lb.resize_cooldown, is_oscillation ? ", oscillation" : "");

	}

	MAL_TRACE_RESIZE(g.sync.compute_epoch.load(std::memory_order_acquire), old_a_size_, target_);
	g.timing.resize_commit += commit_elapsed;
	g.timing.resize_count++;

}

ResizeDecision run_local_resize_decision(const EpochMetrics& m) {

	ResizeDecision decision;

	if (m.global_remaining < kEpsDone || m.active_n <= 0) {

		decision.done = true;
		return decision;

	}

	const double min_horizon_epochs = g.cfg.resize_min_horizon_epochs.load(std::memory_order_relaxed);

	if (min_horizon_epochs > 0.0 && m.global_thr > kEpsThroughput) {

		const double epoch_secs = std::max(kEpsElapsed, g.cfg.epoch_ms.load(std::memory_order_relaxed) / 1000.0);

		double remaining_time = m.global_remaining / m.global_thr;

		if (m.min_thr > kEpsThroughput && m.active_n > 0) {

			const double straggler_time = (m.global_remaining / (double)m.active_n) / m.min_thr;
			remaining_time = std::max(remaining_time, straggler_time);

		}

		if (remaining_time < min_horizon_epochs * epoch_secs) {

			decision.vote = MAL_VOTE_KEEP;
			decision.target_active_size = -1;
			return decision;

		}

	}

	decision = g.cfg.decide_resize_func(m);

	if (decision.vote != MAL_VOTE_KEEP && decision.vote != MAL_VOTE_RESIZE && decision.vote != MAL_VOTE_ABSTAIN) {

		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Decision returned invalid vote=%d, abstaining", (int)decision.vote);
		decision.vote = MAL_VOTE_ABSTAIN;

	}

	if (decision.vote == MAL_VOTE_RESIZE && decision.target_active_size <= 0) {

		MAL_LOG_L(MAL_LOG_WARN, "EPOCH", "Decision returned invalid target=%d (valid range 1..%d), abstaining", decision.target_active_size, g.comm.u_size);
		decision.vote = MAL_VOTE_ABSTAIN;

	}

	g.lb.last_decision_settled = decision.vote != MAL_VOTE_ABSTAIN && decision.settled;

	if (decision.vote != MAL_VOTE_RESIZE) {

		decision.target_active_size = -1;
		return decision;

	}

	decision.target_active_size = std::min(decision.target_active_size, g.comm.u_size);
	return decision;

}

struct ResizeConsensus {

	bool should_resize{false};
	int target{-1};
	int active_size{-1};
	unsigned long long local_decision_epoch{0};

};

inline long long quorum_count(double fraction, long long n) {

	return (long long)std::ceil(fraction * (double)n - 1e-9);

}

ResizeConsensus resize_consensus() {

	const double t_decision_start = MPI_Wtime();

	ResizeConsensus out;

	EpochMetrics m = gather_epoch_metrics();

	const long long local_finalize = g.sync.finalize_requested.load(std::memory_order_acquire) ? 1LL : 0LL;
	long long any_finalize_probe = 0;
	MPI_Allreduce(&local_finalize, &any_finalize_probe, 1, MPI_LONG_LONG, MPI_MAX, g.comm.universe);
	if (any_finalize_probe) {
		g.sync.stop.store(true, std::memory_order_release);
		g.sync.notify();
		return out;
	}

	if (!m.any_has_loop) {

		g.timing.epoch_decision += MPI_Wtime() - t_decision_start;
		g.timing.epoch_decision_count++;
		return out;

	}

	const unsigned long long pre_decision_epoch = g.sync.compute_epoch.load(std::memory_order_acquire);
	const bool is_active = (g.comm.active != MPI_COMM_NULL);
	ResizeDecision local_decision = is_active ? run_local_resize_decision(m) : ResizeDecision{};
	const unsigned long long post_decision_epoch = g.sync.compute_epoch.load(std::memory_order_acquire);
	const unsigned long long decision_epoch = std::max(pre_decision_epoch, post_decision_epoch);
	out.local_decision_epoch = decision_epoch;

	if (g.comm.u_rank == 0) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AUTO", "Distributed resize evaluation: active=%d universe=%d epoch=%llu", g.comm.a_size, g.comm.u_size, decision_epoch);

	}

	const long long local_active = (long long)g.comm.a_size;
	const long long local_done = local_decision.done ? 1LL : 0LL;
	const long long local_gen = (long long)g.loop_gen.load(std::memory_order_acquire);
	const long long neutral = LLONG_MIN / 2;
	const long long local_vote = (long long)local_decision.vote;
	const long long local_target = (long long)local_decision.target_active_size;

	long long reduce_in[9] = {local_active, local_finalize, local_done, local_gen, -local_gen, is_active ? local_vote : neutral, is_active ? -local_vote : neutral, is_active ? local_target : neutral, is_active ? -local_target : neutral};
	long long reduce_out[9] = {};

	MPI_Allreduce(reduce_in, reduce_out, 9, MPI_LONG_LONG, MPI_MAX, g.comm.universe);

	const long long any_finalize = reduce_out[1];
	const long long any_done = reduce_out[2];
	const long long max_gen = reduce_out[3];
	const long long min_gen = -reduce_out[4];
	const long long max_vote = reduce_out[5];
	const long long min_vote = -reduce_out[6];
	const long long max_target = reduce_out[7];
	const long long min_target = -reduce_out[8];

	out.active_size = (int)reduce_out[0];

	if (any_finalize) {

		g.sync.stop.store(true, std::memory_order_release);
		g.sync.notify();
		return out;

	}

	if (any_done || min_gen != max_gen) {

		if (any_done && min_gen == max_gen && max_gen > 0) {

			g.sync.loop_done_gen.store((unsigned long long)max_gen, std::memory_order_release);

		}

		g.sync.notify();
		return out;

	}

	if (max_vote == min_vote && max_target == min_target) {

		out.should_resize = (max_vote == MAL_VOTE_RESIZE);
		out.target = out.should_resize ? (int)max_target : -1;

		if (g.comm.u_rank == 0) {

			MAL_LOG_L(MAL_LOG_DEBUG, "VOTE", "unanimous vote=%lld target=%lld -> should=%d target=%d", max_vote, max_target, (int)out.should_resize, out.target);

		}

		g.timing.epoch_decision += MPI_Wtime() - t_decision_start;
		g.timing.epoch_decision_count++;
		return out;

	}

	const int U = g.comm.u_size;
	const int active = out.active_size;
	const size_t abstain_slot = (size_t)U + 1;

	std::vector<long long>& hist = g.lb.vote_hist;
	hist.assign((size_t)U + 2, 0LL);

	if (is_active) {

		if (local_decision.vote == MAL_VOTE_KEEP) {

			hist[0] = 1;

		} else if (local_decision.vote == MAL_VOTE_ABSTAIN) {

			hist[abstain_slot] = 1;

		} else {

			hist[(size_t)local_decision.target_active_size] = 1;

		}

	}

	MPI_Allreduce(MPI_IN_PLACE, hist.data(), U + 2, MPI_LONG_LONG, MPI_SUM, g.comm.universe);

	long long grow = 0;
	long long shrink = 0;
	long long rebalance = 0;

	for (int t = 1; t <= U; t++) {

		if (t < active) shrink += hist[(size_t)t];
		else if (t > active) grow += hist[(size_t)t];
		else rebalance += hist[(size_t)t];

	}

	const long long keep = hist[0];
	const long long abstain = hist[abstain_slot];
	const long long voters = keep + grow + shrink + rebalance;
	const double quorum = g.cfg.resize_quorum.load(std::memory_order_relaxed);
	const long long need = std::max(voters / 2 + 1, quorum_count(quorum, voters));

	int lo = 1;
	int hi = 0;
	long long count = 0;

	if (voters > 0 && voters >= quorum_count(kMinVoteTurnout, voters + abstain)) {

		if (grow >= need) {

			lo = active + 1;
			hi = U;
			count = grow;

		} else if (shrink >= need) {

			lo = 1;
			hi = active - 1;
			count = shrink;

		} else if (rebalance >= need) {

			lo = active;
			hi = active;
			count = rebalance;

		}

	}

	if (count > 0) {

		const long long median_pos = (lo > active) ? (count + 1) / 2 : count / 2 + 1;
		long long seen = 0;

		for (int t = lo; t <= hi; t++) {

			seen += hist[(size_t)t];

			if (seen >= median_pos) {

				out.target = t;
				break;

			}

		}

		out.should_resize = true;

	}

	if (g.comm.u_rank == 0) {

		MAL_LOG_L(MAL_LOG_DEBUG, "VOTE", "keep=%lld grow=%lld shrink=%lld rebalance=%lld abstain=%lld need=%lld quorum=%.2f -> should=%d target=%d", keep, grow, shrink, rebalance, abstain, need, quorum, (int)out.should_resize, out.target);

	}

	g.timing.epoch_decision += MPI_Wtime() - t_decision_start;
	g.timing.epoch_decision_count++;

	return out;

}

static void sync_decide_resize_state_after_commit() {

	if (g.comm.active == MPI_COMM_NULL || g.comm.a_size <= 1) return;
	if (!g.cfg.decide_resize_state_save || !g.cfg.decide_resize_state_load) return;

	unsigned long len = 0;
	std::vector<unsigned char> buf;

	if (g.comm.a_rank == 0) {

		ResizeStateBlob b = g.cfg.decide_resize_state_save();

		if (b.len > (size_t)INT_MAX) {

			MAL_LOG_L(MAL_LOG_ERROR, "STATE", "decide_resize_state_save() returned len=%zu, too large for MPI int count", b.len);
			MPI_Abort(g.comm.active, 1);

		}

		len = (unsigned long)b.len;
		buf.assign((const unsigned char*)b.data, (const unsigned char*)b.data + b.len);

	}

	MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, 0, g.comm.active);
	buf.resize(len);

	if (len > 0) {

		MPI_Bcast(buf.data(), (int)len, MPI_BYTE, 0, g.comm.active);

	}

	g.cfg.decide_resize_state_load(buf.data(), (size_t)len);

}

bool prepare_resize_if_needed() {

	if (!g.cfg.malleability_enabled.load(std::memory_order_relaxed)) {

		return false;

	}

	if (!g.cfg.enabled.load(std::memory_order_relaxed) && !g.cfg.load_balancing_enabled.load(std::memory_order_relaxed)) {

		if (g.sync.finalize_requested.load(std::memory_order_acquire)) {

			g.sync.stop.store(true, std::memory_order_release);
			g.sync.notify();

		}

		return false;

	}

	ResizeConsensus consensus = resize_consensus();
	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Consensus: should=%d target=%d active=%d", (int)consensus.should_resize, consensus.target, consensus.active_size);

	if (!consensus.should_resize) {

		return false;

	}

	if (g.sync.stop.load(std::memory_order_acquire)) {

		return false;

	}

	Resizer resizer(consensus.target);

	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Committing resize target=%d", consensus.target);

	resizer.prepare_phase();

	g.sync.resize_pending.store(true, std::memory_order_release);
	g.sync.notify();

	resizer.commit_phase();

	g.sync.resize_pending.store(false, std::memory_order_release);

	sync_decide_resize_state_after_commit();

	MAL_LOG_L(MAL_LOG_DEBUG, "EPOCH", "Commit complete (active=%d)", g.comm.a_size);

	g.sync.notify();

	return true;

}

void clear_prepared_resize() {

	std::lock_guard lk(g.resize_mu);
	g.prepared_resize.reset();
	g.prepared_resize_ready.store(false, std::memory_order_release);

}

inline int effective_epoch_interval_ms() {

	const int wait_ms = g.cfg.epoch_ms.load(std::memory_order_relaxed);
	return wait_ms > 0 ? wait_ms : kDefaultEpochIntervalMs;

}

static void worker_self_sample();

inline bool process_step_request(std::chrono::steady_clock::time_point& next_step_decision, const std::chrono::steady_clock::time_point& start_tp, bool needs_initial_rampup) {

	if (!g.sync.step_request.load(std::memory_order_acquire)) {

		return false;

	}

	const auto now = std::chrono::steady_clock::now();
	const bool finalize_local = g.sync.finalize_requested.load(std::memory_order_acquire);
	const long long last_step_local = (g.sync.iter_horizon.load(std::memory_order_acquire) <= 1) ? 1LL : 0LL;
	long long want_eval_local = 0;

	if (g.sync.step_force_eval.load(std::memory_order_acquire)) {

		want_eval_local = 1;

	} else if (g.comm.u_rank == 0) {

		want_eval_local = (now >= next_step_decision) ? 1 : 0;

	}

	long long red_in[3] = { want_eval_local, finalize_local ? 1LL : 0LL, -last_step_local };
	long long red_out[3] = { 0, 0, 0 };
	MPI_Allreduce(red_in, red_out, 3, MPI_LONG_LONG, MPI_MAX, g.comm.universe);

	const bool do_eval = (red_out[0] != 0);
	const bool any_finalize = (red_out[1] != 0);
	const bool all_last_step = (-red_out[2] != 0);

	if (g.loop != nullptr && g.sync.step_buf != nullptr) {

		if (all_last_step || any_finalize) {

			mal_allgather_replicated(*g.loop, g.sync.step_buf, g.sync.step_elem, g.sync.step_total_n);

		} else {

			halo_exchange_field(*g.loop, g.sync.step_buf, g.sync.step_elem, g.sync.step_total_n);

		}

	}

	bool committed = false;

	if (any_finalize || all_last_step) {

		g.sync.stop.store(true, std::memory_order_release);

	} else if (do_eval) {

		committed = prepare_resize_if_needed();
		worker_self_sample();

		if (g.comm.u_rank == 0) {

			const int epoch_ms = effective_epoch_interval_ms();
			next_step_decision = std::chrono::steady_clock::now() + std::chrono::milliseconds(epoch_ms);

		}

		if (g.comm.active != MPI_COMM_NULL) {

			g.lb.epoch_assigned = 0;
			g.lb.epoch_start_time = MPI_Wtime();

		}

	}

	if (committed && g.lb.last_commit_grew && g.comm.active != MPI_COMM_NULL && g.comm.a_size > 1 && g.sync.step_buf != nullptr && g.sync.step_total_n > 0) {

		const long bytes = g.sync.step_total_n * (long)g.sync.step_elem;

		if (bytes <= INT_MAX) {

			int rc = MPI_Bcast(g.sync.step_buf, (int)bytes, MPI_BYTE, 0, g.comm.active);

			if (rc != MPI_SUCCESS) {

				char err[MPI_MAX_ERROR_STRING] = {};
				int len = 0;
				MPI_Error_string(rc, err, &len);
				MAL_LOG_L(MAL_LOG_ERROR, "STEP", "MPI_Bcast(field refresh) failed rc=%d msg=%.*s", rc, len, err);

			}

		} else {

			MAL_LOG_L(MAL_LOG_ERROR, "STEP", "field refresh too large for MPI int count (bytes=%ld)", bytes);
			MPI_Abort(g.comm.active, 1);

		}

	}

	(void)start_tp; (void)needs_initial_rampup; (void)committed;

	g.sync.step_buf = nullptr;
	g.sync.step_request.store(false, std::memory_order_release);
	g.sync.step_done.store(true, std::memory_order_release);
	g.sync.notify();

	return true;

}

static void worker_self_sample() {

	#ifdef __linux__

		struct timespec ts;

		if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {

			g.worker_cpu_ns.store((long long)ts.tv_sec * 1000000000LL + ts.tv_nsec, std::memory_order_release);

		}

		FILE* f = std::fopen("/proc/thread-self/schedstat", "r");

		if (f != nullptr) {

			unsigned long long on_cpu = 0, run_delay = 0;

			if (std::fscanf(f, "%llu %llu", &on_cpu, &run_delay) == 2) {

				g.worker_runq_ns.store((long long)run_delay, std::memory_order_release);

			}

			std::fclose(f);

		}

		g.worker_last_cpu.store(sched_getcpu(), std::memory_order_release);

	#endif

}

void progress_thread() {

	#ifdef __linux__

		g.worker_tid.store((long)syscall(SYS_gettid), std::memory_order_release);

	#endif

	worker_self_sample();

	#ifdef __APPLE__

		if (g.cfg.affinity_enabled) {

			#if defined(__arm64__) || defined(__aarch64__)

				pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
				MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "worker: QoS set to E-Core");

			#elif defined(__x86_64__) || defined(__i386__)

				mach_port_t self = mach_thread_self();
				thread_affinity_policy_data_t policy = { 1 };
				thread_policy_set(self, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
				mach_port_deallocate(mach_task_self(), self);
				MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "worker: affinity hint set to E-Core");

			#endif

		}

	#endif

	std::vector<std::function<void()>> batch;
	const bool needs_initial_rampup = g.comm.a_size > 0 && g.comm.a_size < g.comm.u_size && g.cfg.resize_policy != MAL_RESIZE_POLICY_FIXED_SEQUENCE;
	const auto worker_start_tp = std::chrono::steady_clock::now();
	auto next_resize_check = worker_start_tp + std::chrono::milliseconds(needs_initial_rampup ? 1 : effective_epoch_interval_ms());

	auto next_step_decision = worker_start_tp + std::chrono::milliseconds(needs_initial_rampup ? 1 : effective_epoch_interval_ms());

	bool iterative_mode = false;

	const bool worker_immutable = !g.cfg.malleability_enabled.load(std::memory_order_relaxed) || (!g.cfg.enabled.load(std::memory_order_relaxed) && !g.cfg.load_balancing_enabled.load(std::memory_order_relaxed));

	while (!g.sync.stop.load(std::memory_order_relaxed)) {

		worker_self_sample();

		if (g.sync.attach_pending.load(std::memory_order_acquire)) {

			for (;;) {

				{

					std::lock_guard lk(g.attach_mu);

					if (g.attach_tasks.empty()) {

						g.sync.attach_pending.store(false, std::memory_order_release);
						g.sync.notify();
						break;

					}

					batch.swap(g.attach_tasks);

				}

				for (auto& fn : batch) {

					if (fn) {

						fn();

					}

				}

				batch.clear();

			}

		}

		if (g.sync.step_request.load(std::memory_order_acquire)) {

			iterative_mode = true;
			process_step_request(next_step_decision, worker_start_tp, needs_initial_rampup);

			if (g.sync.stop.load(std::memory_order_relaxed)) {

				break;

			}

			continue;

		}

		if (worker_immutable) {

			std::unique_lock lk(g.sync.mu);
			g.sync.cv.wait(lk, [] {

				return g.sync.stop.load(std::memory_order_relaxed) || g.sync.finalize_requested.load(std::memory_order_relaxed) || g.sync.attach_pending.load(std::memory_order_relaxed) || g.sync.step_request.load(std::memory_order_relaxed);

			});

			if (g.sync.finalize_requested.load(std::memory_order_acquire) && !g.sync.stop.load(std::memory_order_acquire)) {

				g.sync.stop.store(true, std::memory_order_release);

			}

			continue;

		}

		const int epoch_snapshot_ms = effective_epoch_interval_ms();

		{

			std::unique_lock lk(g.sync.mu);

			for (;;) {

				const bool should_wake = g.sync.stop.load(std::memory_order_relaxed) || g.sync.finalize_requested.load(std::memory_order_relaxed) || g.sync.attach_pending.load(std::memory_order_relaxed) || g.sync.step_request.load(std::memory_order_relaxed) || effective_epoch_interval_ms() != epoch_snapshot_ms;

				if (should_wake) {

					break;

				}

				if (g.sync.cv.wait_until(lk, next_resize_check) == std::cv_status::timeout) {

					break;

				}

			}

		}

		if (g.sync.stop.load(std::memory_order_relaxed)) {

			break;

		}

		if (g.sync.step_request.load(std::memory_order_acquire)) {

			continue;

		}

		if (g.sync.attach_pending.load(std::memory_order_acquire)) {

			continue;

		}

		const int epoch_ms = effective_epoch_interval_ms();

		if (epoch_ms != epoch_snapshot_ms) {

			next_resize_check = std::chrono::steady_clock::now() + std::chrono::milliseconds(epoch_ms);
			continue;

		}

		const bool finalize_now = g.sync.finalize_requested.load(std::memory_order_acquire);
		const auto now = std::chrono::steady_clock::now();

		if (!finalize_now && now < next_resize_check) {

			continue;

		}

		const bool iterative_now = iterative_mode || g.sync.iterative_kernel.load(std::memory_order_acquire);

		if (iterative_now) {

			next_resize_check = now + std::chrono::milliseconds(std::max(epoch_ms, 1));
			continue;

		}

		const bool should_try_prepare = finalize_now || now >= next_resize_check;

		if (should_try_prepare) {

			(void)prepare_resize_if_needed();

			worker_self_sample();

			next_resize_check = now + std::chrono::milliseconds(epoch_ms);

		}

		if (g.sync.stop.load(std::memory_order_acquire)) {

			break;

		}

		g.sync.notify();

	}

	worker_self_sample();
	g.sync.notify();

}

#endif
