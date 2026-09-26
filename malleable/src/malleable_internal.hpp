#pragma once

#include "malleable.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cfloat>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <dlfcn.h>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <pthread.h>
#include <sstream>
#include <strings.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifdef __linux__

	#include <sched.h>
	#include <sys/syscall.h>

#endif

#ifdef __linux__

	#include <filesystem>
	#include <fstream>
	#include <sched.h>

#endif

#ifdef __APPLE__

	#include <mach/mach.h>
	#include <mach/thread_policy.h>

#endif

#define MAL_ALWAYS_INLINE __attribute__((always_inline)) inline
#define MAL_LIKELY(x) __builtin_expect(!!(x), 1)
#define MAL_UNLIKELY(x) __builtin_expect(!!(x), 0)

bool mal_should_trace();
void mal_trace_start();
void mal_trace_end();
void mal_trace_timer(int rank, const char* name, double seconds);
void mal_trace_resize(long epoch, int old_active, int new_active);
void mal_trace_probe(long epoch, int probe, int active, double throughput, double throughput1, double speedup, double efficiency);

struct MalScopeTimer {

	MalScopeTimer(const char* key);
	~MalScopeTimer();

private:
	const char* key_ = nullptr;
	double start_t_ = 0.0;
};

template<typename T> inline void mal_trace_meta(const char* key, const T& value) {
	if (mal_should_trace()) {
		std::ostringstream ss;
		ss << value;
		std::printf("META,%d,%s,%s\n", mal_rank(), key, ss.str().c_str());
	}
}
template<typename T> inline void mal_trace_result(const char* key, const T& value) {
	if (mal_should_trace()) {
		std::ostringstream ss;
		ss << value;
		std::printf("RESULT,%d,%s,%s\n", mal_rank(), key, ss.str().c_str());
	}
}

#if !defined(MAL_TRACE_DISABLED)
#define MAL_TRACE_START() mal_trace_start()
#define MAL_TRACE_END() mal_trace_end()
#define MAL_TRACE_TIMER(key, seconds) mal_trace_timer(mal_rank(), key, seconds)
#define MAL_TRACE_RESIZE(epoch, old, new) mal_trace_resize(epoch, old, new)
#define MAL_TRACE_PROBE(epoch, probe, active, thr, thr1, speedup, efficiency) mal_trace_probe(epoch, probe, active, thr, thr1, speedup, efficiency)
#define MAL_TRACE_META(key, value) mal_trace_meta(key, value)
#define MAL_TRACE_RESULT(key, value) mal_trace_result(key, value)
#else
#define MAL_TRACE_START() ((void)0)
#define MAL_TRACE_END() ((void)0)
#define MAL_TRACE_TIMER(key, seconds) ((void)0)
#define MAL_TRACE_RESIZE(epoch, old, new) ((void)0)
#define MAL_TRACE_PROBE(epoch, probe, active, thr, thr1, speedup, efficiency) ((void)0)
#define MAL_TRACE_META(key, value) ((void)0)
#define MAL_TRACE_RESULT(key, value) ((void)0)
#endif

void mal_set_shared_mem(void* mem);
void* mal_get_shared_mem();

void mal_set_epoch_interval_ms(int ms);
void mal_set_resize_quorum(double quorum);

void mal_set_attach_exec_mode(MalAttachExecMode mode);
[[nodiscard]] MalAttachExecMode mal_get_attach_exec_mode();
void mal_wait_attach_tasks();

[[nodiscard]] MalCollapseSpec mal_make_collapse_spec(const long* extents, size_t ndims);
[[nodiscard]] MalFor mal_for_collapse(const MalCollapseSpec& spec, long* iter, long* limit);
void mal_collapse_decode(const MalCollapseSpec& spec, long flat_iter, long* indices_out);

[[nodiscard]] bool mal_for_nd_done(const MalForND& f);

void mal_attach_vec_ragged(MalFor& f, void** user_ptr, size_t elem_size, long total_inner, const long* row_offsets, long n_rows, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_ONLY);

void mal_allgather_replicated(MalFor& f, void* full_buf, size_t elem_size, long total_n);
void halo_exchange_field(MalFor& f, void* buf, size_t elem, long total);
void mal_step_sync(MalFor& f, void* full_buf, size_t elem_size, long total_n);

enum EpochChangeMode {

	MAL_EPOCH_CHANGE_RECALCULATE = 0,
	MAL_EPOCH_CHANGE_USE_LAST_DECISION = 1,
};

constexpr int kDefaultInitialSize = INT_MAX;
constexpr int kDefaultEpochIntervalMs = 1000;
constexpr int kDefaultEpochChangeMode = MAL_EPOCH_CHANGE_USE_LAST_DECISION;
constexpr MalLogLevel kDefaultLogLevel = MAL_LOG_INFO;
constexpr bool kDefaultLogAllRanks = false;
constexpr bool kDefaultAffinityEnabled = true;
constexpr int kDefaultMainCore = -1;
constexpr int kDefaultWorkerCore = -1;
constexpr bool kDefaultMalleabilityEnabled = false;
constexpr bool kDefaultLoadBalancingEnabled = false;

class Resizer;

struct TransferPlanEntry {

	int old_rank{0};
	int new_rank{0};
	long v_start{0};
	long v_count{0};
};

class BufferPool {

	static constexpr int kTLCacheSlots = 16;

public:
	static constexpr size_t kSmallAllocThreshold = 256;

private:
	struct Entry {

		void* ptr;
		size_t capacity;
	};

	std::unordered_map<size_t, std::vector<Entry>> buckets_;
	std::mutex mtx_;

	struct TLSlot {

		void* ptr{nullptr};
		size_t cap{0};
	};

	struct TLBucket {

		TLSlot slots[kTLCacheSlots]{};
		int count{0};
	};

	using TLMap = std::unordered_map<size_t, TLBucket>;

	struct TLGuard {

		TLMap map;

		~TLGuard() {

			for (auto& [key, tb] : map) {

				for (int i = 0; i < tb.count; i++) {

					if (tb.slots[i].ptr) {

						std::free(tb.slots[i].ptr);
					}
				}
			}
		}
	};

	static TLMap& tl_map() noexcept {

		static thread_local TLGuard g;
		return g.map;
	}

	static unsigned clz64(unsigned long long v) noexcept {

		return (unsigned)std::countl_zero(v);
	}

public:

	static size_t bucket_key(size_t bytes) noexcept {

		if (bytes <= 1) {

			return 1;
		}

		return size_t{1} << (64 - clz64((unsigned long long)(bytes - 1)));
	}

	~BufferPool() {

		std::lock_guard lk(mtx_);

		for (auto& [key, entries] : buckets_) {

			for (auto& e : entries) {

				std::free(e.ptr);
			}
		}
	}

	void* acquire(size_t min_bytes) {

		if (min_bytes <= kSmallAllocThreshold) {

			void* p = std::malloc(min_bytes > 0 ? min_bytes : 1);

			if (MAL_UNLIKELY(!p)) {

				throw std::bad_alloc();
			}

			return p;
		}

		size_t key = bucket_key(min_bytes);

		TLMap& tlm = tl_map();
		auto tlit = tlm.find(key);

		if (MAL_LIKELY(tlit != tlm.end())) {

			TLBucket& tb = tlit->second;

			if (tb.count > 0) {

				return tb.slots[--tb.count].ptr;
			}
		}

		{

			std::lock_guard lk(mtx_);
			auto it = buckets_.find(key);

			if (it != buckets_.end() && !it->second.empty()) {

				Entry e = it->second.back();
				it->second.pop_back();

				return e.ptr;
			}
		}

		void* p = std::malloc(key);

		if (MAL_UNLIKELY(!p)) {

			throw std::bad_alloc();
		}

		return p;
	}

	void release(void* ptr, size_t capacity) {

		if (!ptr) {

			return;
		}

		if (capacity <= kSmallAllocThreshold) {

			std::free(ptr);

			return;
		}

		size_t key = bucket_key(capacity);
		TLBucket& tb = tl_map()[key];

		if (MAL_LIKELY(tb.count < kTLCacheSlots)) {

			tb.slots[tb.count++] = {ptr, capacity};
			return;
		}

		std::lock_guard lk(mtx_);
		buckets_[key].push_back({ptr, capacity});
	}
};

extern BufferPool g_buffer_pool;

struct alignas(64) MalVec {

	void* buf{nullptr};
	void** user_ptr{nullptr};
	long buf_global_start{0};
	long local_n{0};
	long done_n{0};
	size_t elem_size{0};
	size_t buf_bytes{0};
	long total_n{0};

	long plan_origin_n{0};
	void* result_buf{nullptr};
	long cache_start{0};
	long cache_end{0};
	long cache_local_off{0};
	int gather_root{-1};
	MalAttachPolicy attach_policy{MAL_ATTACH_PARTITIONED};
	MalDataAccessMode access_mode{MAL_ACCESS_READ_WRITE};
	bool cache_valid{false};
	bool sealed{false};

	bool ragged{false};
	const long* ragged_row_offsets{nullptr};
	long ragged_n_rows{0};
	void* ragged_full_src{nullptr};
	size_t ragged_full_bytes{0};
	std::vector<long> ragged_bases;

	std::vector<std::pair<long, long>> done_segs;

	MAL_ALWAYS_INLINE void sync_user_ptr() noexcept {

		if (MAL_UNLIKELY(!user_ptr)) {

			return;
		}

		*user_ptr = static_cast<char*>(buf) - buf_global_start * (long)elem_size;
	}

	void free_resources();
};

struct MalAcc {

	void* ptr{nullptr};
	void (*fn_get)(const void*, void*){nullptr};
	void (*fn_set)(void*, const void*){nullptr};
	void (*fn_add)(void*, const void*){nullptr};
	void (*fn_reset)(void*){nullptr};
	size_t esz{sizeof(long)};

	std::vector<char> epoch_buf;
	std::vector<char> shadow;
	std::vector<char> capture;
	long shadow_iter{LONG_MIN};
	bool capture_valid{false};
	bool needs_reset{false};
	bool sealed{false};

	int result_rank{0};
	int dtype_idx{1};
	int dop_idx{0};
};

struct SharedMat {

	void* buf{nullptr};
	size_t total_bytes{0};
	bool user_owned{false};
	void** user_ptr{nullptr};

	void free_resources();
};

struct StagedBuffer {

	void* ptr{nullptr};
	size_t bytes{0};
};

struct PendingActivation {

	std::vector<std::pair<long, long>> ranges;
	std::vector<StagedBuffer> vec_slices;
	std::vector<std::vector<char>> acc_epoch_bufs;
	std::vector<StagedBuffer> shared_mats;
	std::vector<StagedBuffer> shared_vecs;
	size_t next_acc{0};
	size_t next_shared{0};
	size_t next_shared_vec{0};

	~PendingActivation() {

		for (const auto& buf : vec_slices) {

			g_buffer_pool.release(buf.ptr, buf.bytes);
		}

		for (const auto& buf : shared_mats) {

			g_buffer_pool.release(buf.ptr, buf.bytes);
		}

		for (const auto& buf : shared_vecs) {

			g_buffer_pool.release(buf.ptr, buf.bytes);
		}
	}
};

struct MalState {

	struct CommInfo {

		MPI_Session session{MPI_SESSION_NULL};
		MPI_Group world_group{MPI_GROUP_NULL};
		MPI_Comm universe{MPI_COMM_NULL};
		MPI_Comm app_universe{MPI_COMM_NULL};
		int u_rank{-1};
		int u_size{0};
		MPI_Comm active{MPI_COMM_NULL};
		int a_rank{-1};
		int a_size{0};
		bool active_borrowed{false};

	} comm;

	struct Config {

		MalResizePolicy resize_policy{MAL_RESIZE_POLICY_AUTO};
		DecideResizeFunc decide_resize_func{nullptr};
		void* decide_resize_plugin_handle{nullptr};
		ResizeStateSaveFunc decide_resize_state_save{nullptr};
		ResizeStateLoadFunc decide_resize_state_load{nullptr};
		std::atomic<double> resize_min_horizon_epochs{2.0};
		std::atomic<double> resize_quorum{1.0};
		std::atomic<int> epoch_ms{kDefaultEpochIntervalMs};
		std::atomic<int> epoch_change_mode{kDefaultEpochChangeMode};
		std::atomic<bool> enabled{true};
		std::atomic<MalLogLevel> log_level{kDefaultLogLevel};
		std::atomic<bool> log_all_ranks{kDefaultLogAllRanks};
		std::atomic<bool> malleability_enabled{kDefaultMalleabilityEnabled};
		std::atomic<bool> load_balancing_enabled{kDefaultLoadBalancingEnabled};
		std::atomic<bool> fast_response{false};
		std::atomic<MalAttachExecMode> attach_mode{MAL_ATTACH_SYNC};
		std::atomic<int> stencil_epoch_steps{64};

		std::atomic<int> stencil_resid_reduces{1};

		bool trace_enabled{false};
		bool affinity_enabled{kDefaultAffinityEnabled};
		int main_core{kDefaultMainCore};
		int worker_core{kDefaultWorkerCore};
		int resolved_main_core{-1};
		int resolved_worker_core{-1};
		int node_local_rank{0};
		int initial_size{kDefaultInitialSize};

	} cfg;

	struct SyncBarrier {

		alignas(64) std::mutex plan_mu;
		alignas(64) std::atomic<unsigned long long> loop_done_gen{0};
		alignas(64) std::atomic<bool> compute_ready{false};
		alignas(64) std::atomic<bool> resize_pending{false};
		alignas(64) std::atomic<bool> attach_pending{false};
		alignas(64) std::atomic<bool> stop{false};
		alignas(64) std::atomic<bool> loop_has_new_work{false};
		alignas(64) std::atomic<bool> pending_has_ranges{false};
		alignas(64) std::atomic<unsigned long long> compute_epoch{0};
		alignas(64) std::atomic<bool> finalize_requested{false};
		alignas(64) std::atomic<long> iter_horizon{1};
		alignas(64) std::atomic<bool> iterative_kernel{false};

		alignas(64) std::atomic<bool> step_request{false};
		alignas(64) std::atomic<bool> step_done{false};
		alignas(64) std::atomic<long long> step_counter{0};
		alignas(64) std::atomic<bool> step_force_eval{false};
		alignas(64) std::atomic<long long> eval_stride{0};
		void* step_buf{nullptr};
		size_t step_elem{0};
		long step_total_n{0};

		alignas(64) std::mutex mu;
		std::condition_variable cv;

		template<typename Pred> void compute_wait(Pred ready) {

			if (ready()) {

				return;
			}

			{

				std::lock_guard lk(mu);
				compute_ready.store(true, std::memory_order_release);
			}

			cv.notify_one();

			{

				std::unique_lock lk(mu);
				cv.wait(lk, ready);
				compute_ready.store(false, std::memory_order_release);
			}
		}

		void wait_for_compute() {

			std::unique_lock lk(mu);

			while (!compute_ready.load(std::memory_order_acquire) && !stop.load(std::memory_order_acquire) && !finalize_requested.load(std::memory_order_acquire)) {

				cv.wait(lk);
			}
		}

		void notify() {

			{

				std::lock_guard lk(mu);
			}

			cv.notify_all();
		}

	} sync;

	struct PreparedResize {

		int target{-1};
		unsigned long long local_decision_epoch{0};
		std::unique_ptr<Resizer> work;

		bool ready() const noexcept {

			return work != nullptr;
		}

		void reset() noexcept {

			target = -1;
			local_decision_epoch = 0;
			work.reset();
		}
	};

	struct alignas(64) LoadBalance {

		std::vector<double> weights;
		mutable std::mutex weights_mu;
		double epoch_start_time{0.0};
		long epoch_assigned{0};

		// Last gather_epoch_metrics() call, to measure the throughput of one epoch
		double last_gather_time{0.0};
		double last_gather_start{0.0}; // epoch_start_time at that call
		long last_gather_done{0};

		int resize_cooldown{0};
		int same_size_rebalance_cooldown{0};
		int prev_resize_from{0};
		int prev_resize_to{0};
		bool last_commit_grew{false};

		int my_slow_streak{0};

		bool last_decision_settled{false};

		std::vector<long long> vote_hist;

	} lb;

	std::mutex resize_mu;
	PreparedResize prepared_resize;
	std::atomic<bool> prepared_resize_ready{false};

	std::mutex attach_mu;
	std::vector<std::function<void()>> attach_tasks;

	MalFor* loop{nullptr};

	std::vector<std::unique_ptr<MalVec>> vecs;
	std::vector<std::unique_ptr<MalAcc>> accs;
	std::vector<std::unique_ptr<SharedMat>> shared;
	std::vector<StagedBuffer> gather_cache;

	std::unique_ptr<PendingActivation> pending;

	std::thread worker;
	std::atomic<long> worker_tid{-1};
	std::atomic<unsigned long long> loop_gen{0};
	std::atomic<long long> worker_cpu_ns{-1};
	std::atomic<long long> worker_runq_ns{-1};
	std::atomic<int> worker_last_cpu{-1};

	struct Timing {

		bool enabled{false};
		double t_origin{0.0};
		double init{0.0};
		double mal_for_total{0.0};
		double attach_total{0.0};
		double check_wait_resize{0.0};
		double check_wait_attach{0.0};
		double check_wait_total{0.0};
		double check_for_total{0.0};
		double finalize_worker_join{0.0};
		double finalize_vec_gather{0.0};
		double finalize_acc_reduce{0.0};
		double finalize_cleanup{0.0};
		double resize_prepare{0.0};
		double resize_commit{0.0};
		double epoch_decision{0.0};
		int resize_count{0};
		int epoch_decision_count{0};
		double wait_for_compute{0.0};

	} timing;

	struct {

		void* mem;

	} shared_mem;

	MalState() noexcept = default;
	MalState(const MalState&) = delete;
	MalState& operator=(const MalState&) = delete;
};

extern MalState g;

class Resizer {

	std::vector<std::pair<long, long>> remaining_;
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
	std::vector<std::pair<long, long>> scratch_assigned_;
	std::vector<int> scratch_reuse_flags_;
	std::vector<int> scratch_all_reuse_flags_;

	std::vector<long> flat_buf_;
	std::vector<int> flat_counts_buf_;
	std::vector<int> flat_displs_buf_;
	std::vector<double> all_lb_buf_;
	std::vector<double> fused_gather_buf_;

	std::vector<TransferPlanEntry> build_transfer_plan(const std::vector<long>& old_vs) const;
	void init_vec_tasks(int n, int nvecs, bool was_active);
	void reserve_receiver_buffers(int n, bool am_receiver, const std::vector<TransferPlanEntry>& plan, long my_new_count, const std::vector<int>& all_reuse_flags);
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

inline const MPI_Datatype kDtypeTbl[] = {MPI_INT, MPI_LONG, MPI_LONG_LONG, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_FLOAT, MPI_DOUBLE};

inline const MPI_Op kDopTbl[] = {MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN};

constexpr int kDtypeINT = 0;
constexpr int kDtypeLONG = 1;
constexpr int kDtypeLLONG = 2;
constexpr int kDtypeUINT = 3;
constexpr int kDtypeULONG = 4;
constexpr int kDtypeFLOAT = 5;
constexpr int kDtypeDOUBLE = 6;

constexpr int kDopSUM = 0;
constexpr int kDopPROD = 1;
constexpr int kDopMAX = 2;
constexpr int kDopMIN = 3;

inline MPI_Datatype tag_dtype(int t) noexcept {

	return (t >= 0 && t < (int)std::size(kDtypeTbl)) ? kDtypeTbl[t] : MPI_LONG;
}

inline int dtype_tag(MPI_Datatype d) noexcept {

	for (int i = 0; i < (int)std::size(kDtypeTbl); i++) {

		if (kDtypeTbl[i] == d) {

			return i;
		}
	}

	return 1;
}

inline int dop_tag(MPI_Op d) noexcept {

	for (int i = 0; i < (int)std::size(kDopTbl); i++) {

		if (kDopTbl[i] == d) {

			return i;
		}
	}

	return 0;
}

inline MPI_Op tag_dop(int t) noexcept {

	return (t >= 0 && t < (int)std::size(kDopTbl)) ? kDopTbl[t] : MPI_SUM;
}

int detect_node_local_rank(int fallback_rank);
void pin_main_thread_to_pcore() noexcept;
void pin_worker_thread_to_ecore(std::thread& t) noexcept;

void write_identity(char* dst, int dtype_tag, int dop_tag, int esz);
void combine_with_op(void* dst, const void* src, int dtype_tag, int dop_tag);
void* checked_realloc(void* p, size_t n, const char* ctx);
void pool_reserve(void*& ptr, size_t& capacity, size_t min_bytes, bool preserve_data = true);
void mpi_bcast_bytes(void* buf, size_t bytes, int root, MPI_Comm comm);
void mpi_send_bytes(const void* buf, size_t bytes, int dest, int tag, MPI_Comm comm);
void mpi_recv_bytes(void* buf, size_t bytes, int src, int tag, MPI_Comm comm);

long total_range_iters(const std::vector<std::pair<long, long>>& ranges);
PendingActivation& ensure_pending_activation();
void configure_shared_active_vec(MalVec& v, size_t buf_need);
void release_shared_active_vec(MalVec& v);
void set_partitioned_layout(MalVec& v, long local_n, long plan_origin_n, long buf_global_start);
void install_loop_plan(MalFor& f, const std::vector<std::pair<long, long>>& ranges, const std::vector<long>* local_bases = nullptr);
bool set_read_only_cache_from_ranges(MalVec& v, const std::vector<std::pair<long, long>>& ranges, long local_off);
void refresh_inactive_read_only_cache(MalVec& v);
void advance_read_only_cache_after_progress(MalVec& v, long old_done, long new_done);
StagedBuffer take_pending_vec_slice(int idx);
SharedMat* get_shared_mat_or_abort(int idx);
void run_partitioned_attach_scatter(MalVec& v, void* orig, int result_rank, size_t orig_bytes, MalAttachExecMode exec_mode, std::vector<long> cuts);
void run_shared_active_attach_bcast(MalVec& v, void* orig, size_t total_bytes, MalAttachExecMode exec_mode);
void* acquire_or_broadcast_active_shared_mat(void* orig, size_t total_bytes, MalAttachExecMode exec_mode);
std::vector<char> take_pending_acc_epoch_buf(size_t fallback_size, int dtype_tag, int dop_tag);
StagedBuffer take_pending_shared_mat();
StagedBuffer take_pending_shared_vec();
void load_pending_ranges_into_loop(MalFor& f);
void sync_vec_mapping_for_current_range(MalFor& f);
void append_done_segments(MalVec& v, const MalFor& f, long local_origin, long from_local, long to_local);
void freeze_loop_at_current(MalFor& f);
void distribute(long total, int nprocs, int rank, long& start, long& end) noexcept;
std::vector<long> build_partition_cuts(long total, int nprocs);
void weighted_distribute(long total, int nprocs, int rank, long& vstart, long& vend) noexcept;
std::vector<int> make_displs(const std::vector<int>& counts);
std::vector<std::pair<long, long>> slice_remaining(const std::vector<std::pair<long, long>>& remaining, std::vector<long>& offsets, long vstart, long vend);
bool vec_can_reuse_assigned_ranges(const MalVec& v, const std::vector<std::pair<long, long>>& assigned);
bool vec_reuse_local_copy(MalVec& v, const std::vector<std::pair<long, long>>& assigned, long done_n);

void clear_prepared_resize();
void progress_thread();

void vec_scatter(MalVec& v, const void* root_data, const std::vector<long>& cuts);
void seal_loop_vecs(MalFor& prev);

namespace builtin_auto {
void install(double threshold);
}

namespace builtin_cost {
void install();
}

namespace builtin_fixed_sequence {
void install();
}

inline bool use_async_attach_mode(MalAttachExecMode mode = MAL_ATTACH_INHERIT) {

	MalAttachExecMode effective = mode;

	if (effective == MAL_ATTACH_INHERIT) {

		effective = g.cfg.attach_mode.load();
	}

	return effective == MAL_ATTACH_ASYNC;
}

inline long current_range_local_base(const MalFor& f) {

	return (f.plan_idx < f.plan_local_bases.size()) ? f.plan_local_bases[f.plan_idx] : 0;
}

inline void set_iter(MalFor& f, long v) {

	f.current = v;
	*f.user_iter = v;
}

inline void set_limit(MalFor& f, long v) {

	f.end = v;
	*f.user_limit = v;
}

inline void prime_range_start(MalFor& f) {

	set_iter(f, f.start - 1);
}

template<typename GetAcc, typename OnResult> void batched_allreduce(int n, GetAcc get_acc, OnResult on_result) {

	if (n == 0) {

		return;
	}

	struct Meta {
		int dt, dp, esz;
	};

	std::vector<Meta> meta(n);

	for (int k = 0; k < n; k++) {

		if (MalAcc* a = get_acc(k)) {

			meta[k] = {a->dtype_idx, a->dop_idx, (int)a->esz};

		} else {

			meta[k] = {1, 0, (int)sizeof(long)};
		}
	}

	MPI_Bcast(meta.data(), n * (int)sizeof(Meta), MPI_BYTE, 0, g.comm.universe);

	static thread_local std::vector<char> tl_send, tl_recv;

	int ai = 0;

	while (ai < n) {

		int ae = ai + 1;
		int esz = meta[ai].esz;

		while (ae < n && meta[ae].dt == meta[ai].dt && meta[ae].dp == meta[ai].dp && meta[ae].esz == esz) {

			ae++;
		}

		int gsz = ae - ai;
		size_t total = (size_t)gsz * (size_t)esz;

		tl_send.resize(total);
		tl_recv.resize(total);

		{

			std::lock_guard lk(g.sync.plan_mu);

			for (int k = ai; k < ae; k++) {

				char* slot = tl_send.data() + (k - ai) * esz;
				MalAcc* a = get_acc(k);

				if (a && !a->sealed) {

					if (a->capture_valid && a->capture.size() >= (size_t)esz) {

						std::memcpy(slot, a->capture.data(), (size_t)esz);

					} else if (a->shadow_iter != LONG_MIN && a->shadow.size() >= (size_t)esz) {

						std::memcpy(slot, a->shadow.data(), (size_t)esz);

					} else {

						write_identity(slot, meta[ai].dt, meta[ai].dp, esz);
					}

					if (g.comm.u_rank == 0 && !a->epoch_buf.empty()) {

						combine_with_op(slot, a->epoch_buf.data(), a->dtype_idx, a->dop_idx);
					}

				} else {

					write_identity(slot, meta[ai].dt, meta[ai].dp, esz);
				}
			}
		}

		MPI_Allreduce(tl_send.data(), tl_recv.data(), gsz, tag_dtype(meta[ai].dt), tag_dop(meta[ai].dp), g.comm.universe);

		for (int k = ai; k < ae; k++) {

			on_result(k, tl_recv.data() + (k - ai) * esz, esz);
		}

		ai = ae;
	}
}
