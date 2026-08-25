#ifndef MALLEABLE_TYPES_CPP_INCLUDED
#define MALLEABLE_TYPES_CPP_INCLUDED

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

enum EpochChangeMode {

	MAL_EPOCH_CHANGE_RECALCULATE = 0, MAL_EPOCH_CHANGE_USE_LAST_DECISION = 1, };

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

static BufferPool g_buffer_pool;

struct alignas(64) MalVec {

	void* buf{nullptr};
	void** user_ptr{nullptr};
	long buf_global_start{0};
	long local_n{0};
	long done_n{0};
	size_t elem_size{0};
	size_t buf_bytes{0};
	long total_N{0};

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

	std::vector<std::pair<long,long>> done_segs;

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
	void (*fn_get) (const void*, void*){nullptr};
	void (*fn_set) (void*, const void*){nullptr};
	void (*fn_add) (void*, const void*){nullptr};
	void (*fn_reset)(void*) {nullptr};
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

	std::vector<std::pair<long,long>> ranges;
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
		DecideResizeFunc decide_resize_func{nullptr}; // NOT atomic: must be set before mal_init()
		void*            decide_resize_plugin_handle{nullptr}; // NOT atomic: dlopen handle, closed in mal_finalize()
		ResizeStateSaveFunc decide_resize_state_save{nullptr}; // NOT atomic: must be set before mal_init()
		ResizeStateLoadFunc decide_resize_state_load{nullptr}; // NOT atomic: must be set before mal_init()
		std::atomic<double> resize_min_horizon_epochs{2.0};
		std::atomic<int> epoch_ms{kDefaultEpochIntervalMs};
		std::atomic<int> epoch_change_mode{kDefaultEpochChangeMode};
		std::atomic<bool> enabled{true};
		std::atomic<MalLogLevel> log_level{kDefaultLogLevel};
		std::atomic<bool> log_all_ranks{kDefaultLogAllRanks};
		std::atomic<bool> malleability_enabled{kDefaultMalleabilityEnabled};
		std::atomic<bool> load_balancing_enabled{kDefaultLoadBalancingEnabled};
		std::atomic<bool> fast_response{false};
		std::atomic<double> cost_keep_fraction{0.97};
		std::atomic<bool> baseline_from_perrank{false};
		std::atomic<MalAttachExecMode> attach_mode{MAL_ATTACH_SYNC};
		std::atomic<int> stencil_epoch_steps{64};

		std::atomic<int> stencil_resid_reduces{1};

		std::atomic<int> cost_sample_step{8};

		std::atomic<bool> cost_sample_refine{false};

		std::atomic<int> cost_sample_meas{3};

		bool trace_enabled {false};
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

		int resize_cooldown{0};
		int same_size_rebalance_cooldown{0};
		int prev_resize_from{0};
		int prev_resize_to{0};
		bool last_commit_grew{false};

		int my_slow_streak{0};

		// Set from the last decide_resize_func() call's ResizeDecision.settled,
		// read back into EpochMetrics.any_settled next epoch (see
		// gather_epoch_metrics()). Policy-agnostic: any plugin can report
		// its own convergence this way.
		bool last_decision_settled{false};

		// Set from the last decide_resize_func() call's
		// ResizeDecision.skip_cooldown, consumed once by
		// Resizer::commit_phase() right after the resize it applies to.
		bool last_decision_skip_cooldown{false};

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

	MalState() noexcept = default;
	MalState(const MalState&) = delete;
	MalState& operator=(const MalState&) = delete;

};

static MalState g;

static void seal_loop_vecs(MalFor& prev);

MalFor::~MalFor() {

	phase.store(MAL_LOOP_FINISHED, std::memory_order_release);

	if (g.loop == this) {

		seal_loop_vecs(*this);
		g.loop = nullptr;

	}

}

MalFor::MalFor(MalFor&& other) noexcept : start(other.start), end(other.end), current(other.current), user_iter(other.user_iter), user_limit(other.user_limit), plan_idx(other.plan_idx), check_counter(other.check_counter), gen(other.gen), plan_ranges(std::move(other.plan_ranges)), plan_local_bases(std::move(other.plan_local_bases)), vecs(std::move(other.vecs)), accs(std::move(other.accs)) {

	phase.store(other.phase.load(std::memory_order_relaxed), std::memory_order_relaxed);
	confirmed_iter.store(other.confirmed_iter.load(std::memory_order_relaxed), std::memory_order_relaxed);

	if (g.loop == &other) {

		g.loop = this;

	}

	other.user_iter = nullptr;
	other.user_limit = nullptr;
	other.current = 0;
	other.start = 0;
	other.end = 0;
	other.plan_idx = 0;
	other.check_counter = 0;
	other.phase.store(MAL_LOOP_FINISHED, std::memory_order_release);

}

static int detect_node_local_rank(int fallback_rank) {

	static const char* const kLocalRankVars[] = {

		"OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK", "PMIX_LOCAL_RANK", "PMI_LOCAL_RANK", };

	for (const char* var : kLocalRankVars) {

		if (const char* v = std::getenv(var)) {

			char* end = nullptr;
			long r = std::strtol(v, &end, 10);

			if (end != v && r >= 0) {

				return (int)r;

			}

		}

	}

	return std::max(0, fallback_rank);

}

#ifdef __linux__

	static std::vector<std::pair<int, unsigned long>> linux_get_metrics() {

		std::vector<std::pair<int, unsigned long>> cores;

		try {

			for (const auto& entry : std::filesystem::directory_iterator("/sys/devices/system/cpu/")) {

				const std::string name = entry.path().filename().string();

				if (name.rfind("cpu", 0) != 0 || name.size() <= 3 || !isdigit(name[3])) {

					continue;

				}

				int id;

				try {

					id = std::stoi(name.substr(3));

				} catch (...) {

					continue;

				}

				{

					std::ifstream f(entry.path() / "cpu_capacity");

					if (f.is_open()) {

						unsigned long v = 0;

						f >> v;

						if (f) {

							cores.emplace_back(id, v);
							continue;

						}

					}

				}

				{

					std::ifstream f(entry.path() / "cpufreq/cpuinfo_max_freq");

					if (f.is_open()) {

						unsigned long v = 0;

						f >> v;

						if (f) {

							cores.emplace_back(id, v);
							continue;

						}

					}

				}

			}

		} catch (...) {}

		return cores;

	}

	static const cpu_set_t& linux_allowed_mask() {

		static const cpu_set_t mask = [] {

			cpu_set_t m;
			CPU_ZERO(&m);

			if (sched_getaffinity(0, sizeof(m), &m) != 0) {

				for (int i = 0; i < CPU_SETSIZE; i++) {

					CPU_SET(i, &m);

				}

			}

			return m;

		}();

		return mask;

	}

	static void linux_split_cores(std::vector<int>& pcores, std::vector<int>& ecores) {

		auto cores = linux_get_metrics();

		if (cores.empty()) {

			return;

		}

		const cpu_set_t& allowed = linux_allowed_mask();
		std::vector<std::pair<int, unsigned long>> usable;
		usable.reserve(cores.size());

		for (auto& [id, v] : cores) {

			if (id >= 0 && id < CPU_SETSIZE && CPU_ISSET(id, &allowed)) {

				usable.emplace_back(id, v);

			}

		}

		if (!usable.empty()) {

			cores.swap(usable);

		}

		unsigned long maxv = 0;
		unsigned long minv = ULONG_MAX;

		for (auto& [id, v] : cores) {

			maxv = std::max(maxv, v);
			minv = std::min(minv, v);

		}

		if (maxv == minv) {

			for (auto& [id, _] : cores) {

				pcores.push_back(id);

			}

			return;

		}

		for (auto& [id, v] : cores) {

			if (v == maxv) {

				pcores.push_back(id);

			} else {

				ecores.push_back(id);

			}

		}

	}

	static std::optional<int> linux_pick_core(bool want_pcore, int exclude = -1) {

		std::vector<int> pcores, ecores;
		linux_split_cores(pcores, ecores);

		auto& pool = want_pcore ? pcores : ecores;

		if (pool.empty()) {

			return std::nullopt;

		}

		std::sort(pool.begin(), pool.end());

		const size_t base = (size_t)std::max(0, g.cfg.node_local_rank);

		for (size_t i = 0; i < pool.size(); i++) {

			int core = pool[(base + i) % pool.size()];

			if (core != exclude) {

				return core;

			}

		}

		return std::nullopt;

	}

	static bool linux_pin_thread(pthread_t pt, int core_id, const char* label) noexcept {

		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(core_id, &cpuset);

		int rc = pthread_setaffinity_np(pt, sizeof(cpu_set_t), &cpuset);

		if (rc != 0) {

			MAL_LOG_L(MAL_LOG_WARN, "AFFINITY", "%s: failed to pin to core %d (err=%d)", label, core_id, rc);

			return false;

		}

		MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "%s: pinned to core %d", label, core_id);

		return true;

	}

	static int linux_pin(pthread_t pt, bool want_pcore, int core_cfg, const char* label, int exclude_core = -1) noexcept {

		int core_id = core_cfg;

		if (core_id < 0) {

			if (auto core = linux_pick_core(want_pcore, exclude_core)) {

				core_id = *core;

				MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "%s: auto-selected %s core %d", label, want_pcore ? "P-core" : "E-core", core_id);

			} else {

				MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "%s: no %s core found, not pinning", label, want_pcore ? "P-core" : "E-core");

				return -1;

			}

		} else {

			if (core_id >= CPU_SETSIZE || !CPU_ISSET(core_id, &linux_allowed_mask())) {

				MAL_LOG_L(MAL_LOG_WARN, "AFFINITY", "%s: configured core %d is outside the allowed cpuset (resource manager / launcher binding), not pinning", label, core_id);

				return -1;

			}

			MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "%s: using configured core %d", label, core_id);

		}

		return linux_pin_thread(pt, core_id, label) ? core_id : -1;

	}

#endif

#ifdef __APPLE__

	static void apple_pin(bool is_self, bool want_pcore, const char* label) noexcept {

		#if defined(__arm64__) || defined(__aarch64__)

			if (!is_self) {

				return;

			}

			qos_class_t qos = want_pcore ? QOS_CLASS_USER_INITIATED : QOS_CLASS_UTILITY;

			pthread_set_qos_class_self_np(qos, 0);

			MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "%s: QoS set to %s", label, want_pcore ? "P-core" : "E-core");

		#else

			(void)is_self;
			(void)want_pcore;

			MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "%s: affinity not supported on Intel macOS", label);

		#endif

	}

#endif

void pin_main_thread_to_pcore() noexcept {

	if (!g.cfg.affinity_enabled) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "main: pinning disabled");

		return;

	}

	#ifdef __linux__

		g.cfg.resolved_main_core = linux_pin(pthread_self(), true, g.cfg.main_core, "main");

	#endif

	#ifdef __APPLE__

		apple_pin(true, true, "main");

	#endif

}

void pin_worker_thread_to_ecore(std::thread& t) noexcept {

	if (!g.cfg.affinity_enabled) {

		MAL_LOG_L(MAL_LOG_DEBUG, "AFFINITY", "worker: pinning disabled");

		return;

	}

	#ifdef __linux__

		g.cfg.resolved_worker_core = linux_pin(t.native_handle(), false, g.cfg.worker_core, "worker", g.cfg.resolved_main_core);

	#endif

	#ifdef __APPLE__

		(void)t;
		apple_pin(false, false, "worker");

	#endif

}

int mal_rank() {

	return g.comm.u_rank;

}

int mal_size() {

	return g.comm.u_size;

}

long mal_worker_tid() {

	return g.worker_tid.load(std::memory_order_acquire);

}

int mal_worker_core() {

	if (g.cfg.resolved_worker_core >= 0) {

		return g.cfg.resolved_worker_core;

	}

	return g.worker_last_cpu.load(std::memory_order_acquire);

}

double mal_worker_cpu_seconds() {

	#ifdef __linux__

		if (g.worker.joinable()) {

			clockid_t cid;
			struct timespec ts;

			if (pthread_getcpuclockid(g.worker.native_handle(), &cid) == 0 && clock_gettime(cid, &ts) == 0) {

				return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;

			}

		}

	#endif

	const long long ns = g.worker_cpu_ns.load(std::memory_order_acquire);

	return ns >= 0 ? (double)ns * 1e-9 : -1.0;

}

double mal_worker_runq_seconds() {

	const long long ns = g.worker_runq_ns.load(std::memory_order_acquire);

	return ns >= 0 ? (double)ns * 1e-9 : -1.0;

}

int mal_active_size() {

	return g.comm.a_size;

}

double mal_t_origin() {

	return g.timing.t_origin;

}

bool mal_should_log(MalLogLevel level) {

	if ((int)level < (int)g.cfg.log_level.load(std::memory_order_relaxed)) {

		return false;

	}

	if (g.cfg.log_all_ranks.load(std::memory_order_relaxed)) {

		return true;

	}

	return mal_rank() <= 0;

}

const char* mal_log_level_name(MalLogLevel level) {

	switch (level) {

		case MAL_LOG_DEBUG: return "DEBUG";
		case MAL_LOG_INFO: return "INFO";
		case MAL_LOG_WARN: return "WARN";
		case MAL_LOG_ERROR: return "ERROR";
		case MAL_LOG_NONE: return "NONE";

	}

	return "INFO";

}

bool mal_should_trace() {

	return g.cfg.trace_enabled;

}

void mal_trace_start() {

	if (mal_should_trace())
		std::printf("START,%d,%.9f\n", mal_rank(), MPI_Wtime() - mal_t_origin());

}

void mal_trace_end() {

	if (mal_should_trace()) {

		std::printf("END,%d,%.9f\n", mal_rank(), MPI_Wtime() - mal_t_origin());
		std::fflush(stdout);

	}

}

void mal_trace_timer(int rank, const char* name, double seconds) {

	if (mal_should_trace())
		std::printf("TIMING,%d,%.9f,%s,%.9f\n", rank, MPI_Wtime() - mal_t_origin(), name, seconds);

}

void mal_trace_resize(long epoch, int old_active, int new_active) {

	if (mal_should_trace())
		std::printf("RESIZE,%d,%.9f,%ld,%d,%d\n", mal_rank(), MPI_Wtime() - mal_t_origin(), epoch, old_active, new_active);
}

void mal_trace_probe(long epoch, int probe, int active, double throughput, double throughput1, double speedup, double efficiency) {

	if (mal_should_trace())
		std::printf(
			"PROBE,%d,%.9f,%ld,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
			mal_rank(),
			MPI_Wtime() - mal_t_origin(),
			epoch,
			probe,
			active,
			throughput,
			throughput1,
			speedup,
			efficiency
		);

}

MalScopeTimer::MalScopeTimer(const char* new_key) {

	start_t = MPI_Wtime();
	key = new_key;

}

MalScopeTimer::~MalScopeTimer() {

	MAL_TRACE_TIMER(key, MPI_Wtime() - start_t);

}

const MPI_Datatype kDtypeTbl[] = {MPI_INT, MPI_LONG, MPI_LONG_LONG, MPI_UNSIGNED, MPI_UNSIGNED_LONG, MPI_FLOAT, MPI_DOUBLE};

const MPI_Op kDopTbl[] = {MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN};

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

void write_identity(char* dst, int dtype_tag, int dop_tag, int esz) {

	union { int i; long l; long long ll; unsigned u; unsigned long ul; float f; double d; } v{};

	switch (dop_tag) {

		case kDopSUM: break;

		case kDopPROD:

			switch (dtype_tag) {

				case kDtypeFLOAT: v.f = 1.0f; break;
				case kDtypeDOUBLE: v.d = 1.0; break;
				default: v.l = 1; break;

			} break;

		case kDopMAX:

			switch (dtype_tag) {

				case kDtypeINT: v.i = INT_MIN; break;
				case kDtypeLONG: v.l = LONG_MIN; break;
				case kDtypeLLONG: v.ll = LLONG_MIN; break;
				case kDtypeUINT: v.u = 0; break;
				case kDtypeULONG: v.ul = 0; break;
				case kDtypeFLOAT: v.f = -FLT_MAX; break;
				case kDtypeDOUBLE: v.d = -DBL_MAX; break;

			} break;

		case kDopMIN:

			switch (dtype_tag) {

				case kDtypeINT: v.i = INT_MAX; break;
				case kDtypeLONG: v.l = LONG_MAX; break;
				case kDtypeLLONG: v.ll = LLONG_MAX; break;
				case kDtypeUINT: v.u = UINT_MAX; break;
				case kDtypeULONG: v.ul = ULONG_MAX; break;
				case kDtypeFLOAT: v.f = FLT_MAX; break;
				case kDtypeDOUBLE: v.d = DBL_MAX; break;

			} break;

	}

	std::memcpy(dst, &v, (size_t)esz);

}

template<typename T> inline void combine_with_op_t(void* dst, const void* src, int dop) {

	T& d = *static_cast<T*>(dst);
	const T& s = *static_cast<const T*>(src);

	switch (dop) {

		case kDopSUM: d += s; break;
		case kDopPROD: d *= s; break;
		case kDopMAX: if (s > d) d = s; break;
		case kDopMIN: if (s < d) d = s; break;

	}

}

void combine_with_op(void* dst, const void* src, int dtype_tag, int dop_tag) {

	switch (dtype_tag) {

		case kDtypeINT: combine_with_op_t<int>(dst, src, dop_tag); break;
		case kDtypeLONG: combine_with_op_t<long>(dst, src, dop_tag); break;
		case kDtypeLLONG: combine_with_op_t<long long>(dst, src, dop_tag); break;
		case kDtypeUINT: combine_with_op_t<unsigned>(dst, src, dop_tag); break;
		case kDtypeULONG: combine_with_op_t<unsigned long>(dst, src, dop_tag); break;
		case kDtypeFLOAT: combine_with_op_t<float>(dst, src, dop_tag); break;
		case kDtypeDOUBLE: combine_with_op_t<double>(dst, src, dop_tag); break;

	}

}

void* checked_realloc(void* p, size_t n, const char* ctx) {

	void* nb = std::realloc(p, n);

	if (MAL_UNLIKELY(!nb)) {

		MAL_LOG_L(MAL_LOG_ERROR, "ALLOC", "realloc failed in %s", ctx);
		MPI_Abort(g.comm.universe, 1);

	}

	return nb;

}

void pool_reserve(void*& ptr, size_t& capacity, size_t min_bytes, bool preserve_data = true) {

	size_t need = std::max(size_t{1}, min_bytes);

	if (MAL_LIKELY(ptr && capacity >= need)) {

		return;

	}

	void* nb = g_buffer_pool.acquire(need);
	const size_t new_capacity = (need > BufferPool::kSmallAllocThreshold) ? BufferPool::bucket_key(need) : need;

	if (ptr) {

		if (preserve_data) {

			size_t copy_bytes = std::min(capacity, need);

			if (copy_bytes > 0) {

				std::memcpy(nb, ptr, copy_bytes);

			}

		}

		g_buffer_pool.release(ptr, capacity > 0 ? capacity : 1);

	}

	ptr = nb;
	capacity = new_capacity;

}

void mpi_bcast_bytes(void* buf, size_t bytes, int root, MPI_Comm comm) {

	char* p = static_cast<char*>(buf);
	size_t off = 0;

	while (off < bytes) {

		int chunk = (int)std::min(bytes - off, (size_t)INT_MAX);
		MPI_Bcast(p + off, chunk, MPI_BYTE, root, comm);
		off += (size_t)chunk;

	}

}

void mpi_send_bytes(const void* buf, size_t bytes, int dest, int tag, MPI_Comm comm) {

	const char* p = static_cast<const char*>(buf);
	size_t off = 0;

	while (off < bytes) {

		int chunk = (int)std::min(bytes - off, (size_t)INT_MAX);
		MPI_Send(p + off, chunk, MPI_BYTE, dest, tag, comm);
		off += (size_t)chunk;

	}

}

void mpi_recv_bytes(void* buf, size_t bytes, int src, int tag, MPI_Comm comm) {

	char* p = static_cast<char*>(buf);
	size_t off = 0;

	while (off < bytes) {

		int chunk = (int)std::min(bytes - off, (size_t)INT_MAX);
		MPI_Recv(p + off, chunk, MPI_BYTE, src, tag, comm, MPI_STATUS_IGNORE);
		off += (size_t)chunk;

	}

}

inline bool use_async_attach_mode(MalAttachExecMode mode = MAL_ATTACH_INHERIT) {

	MalAttachExecMode effective = mode;

	if (effective == MAL_ATTACH_INHERIT) {

		effective = g.cfg.attach_mode.load();

	}

	return effective == MAL_ATTACH_ASYNC;

}

void enqueue_attach_task(std::function<void()> fn) {

	{

		std::lock_guard lk(g.attach_mu);
		g.attach_tasks.push_back(std::move(fn));
		g.sync.attach_pending.store(true, std::memory_order_release);

	}

	g.sync.notify();

}

void dispatch_attach_task(std::function<void()> fn, bool async) {

	if (!fn) {

		return;

	}

	if (async) {

		enqueue_attach_task(std::move(fn));
		return;

	}

	fn();

}

void mal_wait_attach_tasks() {

	std::unique_lock lk(g.sync.mu);

	while (g.sync.attach_pending.load(std::memory_order_acquire) && !g.sync.stop.load(std::memory_order_acquire)) {

		g.sync.cv.wait(lk);

	}

}

struct UniverseBroadcastTask {

	void* buf{nullptr};
	size_t bytes{0};

	void operator()() const {

		mpi_bcast_bytes(buf, bytes, 0, g.comm.universe);

	}

};

void run_attach_bcast_once_all(void* buf, size_t bytes, bool wait = true) {

	if (g.comm.universe == MPI_COMM_NULL || bytes == 0) {

		return;

	}

	dispatch_attach_task(UniverseBroadcastTask{buf, bytes}, !wait);

}

inline bool has_work_or_stop() {

	if (g.sync.stop.load(std::memory_order_acquire)) {

		return true;

	}

	if (g.sync.attach_pending.load(std::memory_order_acquire)) {

		return true;

	}

	if (g.sync.loop_has_new_work.load(std::memory_order_acquire)) {

		g.sync.loop_has_new_work.store(false, std::memory_order_relaxed);
		return true;

	}

	return g.sync.pending_has_ranges.load(std::memory_order_acquire);

}

long total_range_iters(const std::vector<std::pair<long,long>>& ranges) {

	long total = 0;

	for (const auto& [a, b] : ranges) {

		total += std::max(0L, b - a);

	}

	return total;

}

std::vector<long> make_range_local_bases(const std::vector<std::pair<long,long>>& ranges) {

	std::vector<long> bases;
	bases.reserve(ranges.size());

	long base = 0;

	for (const auto& [a, b] : ranges) {

		bases.push_back(base);
		base += std::max(0L, b - a);

	}

	return bases;

}

PendingActivation& ensure_pending_activation() {

	if (!g.pending) {

		g.pending = std::make_unique<PendingActivation>();

	}

	return *g.pending;

}

inline bool vec_is_fully_replicated(const MalVec& v) noexcept {

	return v.attach_policy != MAL_ATTACH_PARTITIONED;

}

void configure_shared_active_vec(MalVec& v, size_t buf_need) {

	if (v.buf && v.buf_bytes >= buf_need && v.local_n == v.total_N && v.done_n == 0 && v.buf_global_start == 0 && v.plan_origin_n == 0 && !v.cache_valid) {

		return;

	}

	pool_reserve(v.buf, v.buf_bytes, buf_need);
	v.local_n = v.total_N;
	v.done_n = 0;
	v.buf_global_start = 0;
	v.plan_origin_n = 0;
	v.cache_valid = false;
	v.sync_user_ptr();

}

void release_shared_active_vec(MalVec& v) {

	if (v.buf) {

		g_buffer_pool.release(v.buf, v.buf_bytes > 0 ? v.buf_bytes : 1);
		v.buf = nullptr;
		v.buf_bytes = 0;

	}

	v.local_n = 0;
	v.done_n = 0;
	v.buf_global_start = 0;
	v.plan_origin_n = 0;
	v.cache_valid = false;

	if (v.user_ptr) {

		*v.user_ptr = nullptr;

	}

}

void set_partitioned_layout(MalVec& v, long local_n, long plan_origin_n, long buf_global_start) {

	v.local_n = local_n;
	v.plan_origin_n = plan_origin_n;
	v.buf_global_start = buf_global_start;

}

inline long current_range_local_base(const MalFor& f) {

	return (f.plan_idx < f.plan_local_bases.size()) ? f.plan_local_bases[f.plan_idx] : 0;

}

void install_loop_plan(MalFor& f, const std::vector<std::pair<long,long>>& ranges, const std::vector<long>* local_bases = nullptr) {

	f.plan_ranges = ranges;
	f.plan_local_bases.clear();
	f.plan_idx = 0;

	if (local_bases && local_bases->size() == ranges.size()) {

		f.plan_local_bases = *local_bases;

	} else {

		f.plan_local_bases = make_range_local_bases(ranges);

	}

	if (!f.plan_ranges.empty()) {

		f.start = f.plan_ranges[0].first;
		f.end = f.plan_ranges[0].second;

	} else {

		f.start = 0;
		f.end = 0;

	}

}

bool set_read_only_cache_from_ranges(MalVec& v, const std::vector<std::pair<long,long>>& ranges, long local_off) {

	if (v.access_mode != MAL_ACCESS_READ_ONLY || ranges.empty()) {

		v.cache_valid = false;
		return false;

	}

	long total_len = 0;

	for (const auto& rg : ranges) {

		total_len += std::max(0L, rg.second - rg.first);

	}

	const long cache_start = ranges.front().first;
	const long cache_end = ranges.back().second;

	if (total_len != cache_end - cache_start || cache_end <= cache_start) {

		v.cache_valid = false;
		return false;

	}

	v.cache_valid = true;
	v.cache_start = cache_start;
	v.cache_end = cache_end;
	v.cache_local_off = local_off;
	return true;

}

void refresh_inactive_read_only_cache(MalVec& v) {

	if (v.access_mode != MAL_ACCESS_READ_ONLY) {

		v.cache_valid = false;
		return;

	}

	if (v.cache_valid) {

		v.cache_local_off = std::max(v.cache_local_off, v.done_n);
		return;

	}

	const long cache_start = v.buf_global_start + v.done_n;
	const long cache_end = v.buf_global_start + v.local_n;

	if (v.local_n > v.done_n && cache_start <= cache_end) {

		v.cache_valid = true;
		v.cache_start = cache_start;
		v.cache_end = cache_end;
		v.cache_local_off = v.done_n;
		return;

	}

	v.cache_valid = false;

}

void vec_scatter(MalVec& v, const void* root_data, const std::vector<long>& cuts);

void advance_read_only_cache_after_progress(MalVec& v, long old_done, long new_done) {

	if (v.access_mode != MAL_ACCESS_READ_ONLY || !v.cache_valid || new_done <= old_done) {

		return;

	}

	long delta = new_done - old_done;
	v.cache_start += delta;
	v.cache_local_off += delta;

	if (v.cache_start >= v.cache_end || v.cache_local_off > v.local_n) {

		v.cache_valid = false;

	}

}

StagedBuffer take_pending_vec_slice(int idx) {

	if (!g.pending || idx < 0 || idx >= (int)g.pending->vec_slices.size()) {

		return {};

	}

	StagedBuffer buf = g.pending->vec_slices[(size_t)idx];
	g.pending->vec_slices[(size_t)idx] = {};
	return buf;

}

struct CommBroadcastTask {

	MPI_Comm comm{MPI_COMM_NULL};
	void* buf{nullptr};
	size_t total_bytes{0};

	void operator()() const {

		if (comm == MPI_COMM_NULL || total_bytes == 0) {

			return;

		}

		mpi_bcast_bytes(buf, total_bytes, 0, comm);

	}

};

void async_broadcast_bytes(MPI_Comm comm, void* buf, size_t total_bytes, bool wait) {

	dispatch_attach_task(CommBroadcastTask{comm, buf, total_bytes}, !wait);

}

void init_shared_buffer_from_root(void* buf, size_t total_bytes, bool is_root, const void* orig, const char* warn_msg) {

	if (total_bytes == 0 || !is_root) {

		return;

	}

	if (!orig) {

		MAL_LOG_L(MAL_LOG_WARN, "ATTACH", "%s", warn_msg);
		std::memset(buf, 0, total_bytes);
		return;

	}

	std::memcpy(buf, orig, total_bytes);

}

void maybe_release_root_attach_buffer(void* orig, size_t total_bytes, bool should_release) {

	if (should_release && orig) {

		g_buffer_pool.release(orig, total_bytes > 0 ? total_bytes : 1);

	}

}

SharedMat* get_shared_mat_or_abort(int idx) {

	if (MAL_UNLIKELY(idx < 0 || idx >= (int)g.shared.size() || !g.shared[(size_t)idx])) {

		MAL_LOG_L(MAL_LOG_ERROR, "RESIZE", "Missing shared matrix metadata at index %d", idx);
		MPI_Abort(g.comm.universe, 1);

	}

	return g.shared[(size_t)idx].get();

}

struct PartitionedAttachScatterTask {

	MalVec* vec{nullptr};
	int do_scatter{0};
	void* orig{nullptr};
	int result_rank{-1};
	size_t orig_bytes{0};
	std::vector<long> cuts;

	void operator()() const {

		if (g.comm.active == MPI_COMM_NULL) {

			return;

		}

		int op = do_scatter;
		MPI_Bcast(&op, 1, MPI_INT, 0, g.comm.active);

		if (op) {

			vec_scatter(*vec, orig, cuts);
			maybe_release_root_attach_buffer(orig, orig_bytes, g.comm.u_rank == 0 && result_rank < 0);

		} else if (vec->buf && vec->local_n > 0) {

			std::memset(vec->buf, 0, (size_t)vec->local_n * vec->elem_size);

		}

	}

};

void run_partitioned_attach_scatter(MalVec& v, void* orig, int result_rank, size_t orig_bytes, MalAttachExecMode exec_mode, std::vector<long> cuts) {

	const int do_scatter = (orig && result_rank < 0) ? 1 : 0;

	if (g.comm.a_size == 1) {

		if (do_scatter && orig && v.buf) {

			size_t copy_bytes = (size_t)std::max(0L, v.local_n) * v.elem_size;

			if (copy_bytes > 0) {

				std::memcpy(v.buf, orig, copy_bytes);

			}

		} else if (!do_scatter && v.buf && v.local_n > 0) {

			std::memset(v.buf, 0, (size_t)v.local_n * v.elem_size);

		}

		maybe_release_root_attach_buffer(orig, orig_bytes, g.comm.u_rank == 0 && result_rank < 0);
		return;

	}

	const bool wait = !use_async_attach_mode(exec_mode);
	dispatch_attach_task(PartitionedAttachScatterTask{&v, do_scatter, orig, result_rank, orig_bytes, std::move(cuts)}, !wait);

}

void run_shared_active_attach_bcast(MalVec& v, void* orig, size_t total_bytes, MalAttachExecMode exec_mode) {

	init_shared_buffer_from_root(v.buf, total_bytes, g.comm.a_rank == 0, orig, "MAL_ATTACH_SHARED_ACTIVE vector has null active-root pointer; broadcasting zero-initialized data");
	async_broadcast_bytes(g.comm.active, v.buf, total_bytes, !use_async_attach_mode(exec_mode));
	maybe_release_root_attach_buffer(orig, total_bytes, g.comm.a_rank == 0);

}

void run_shared_all_attach_bcast(void* buf, void* orig, size_t total_bytes, int result_rank, MalAttachExecMode exec_mode, const char* warn_msg) {

	init_shared_buffer_from_root(buf, total_bytes, g.comm.u_rank == 0, orig, warn_msg);
	run_attach_bcast_once_all(buf, total_bytes, !use_async_attach_mode(exec_mode));

	maybe_release_root_attach_buffer(orig, total_bytes, g.comm.u_rank == 0 && result_rank < 0);

}

void* acquire_or_broadcast_active_shared_mat(void* orig, size_t total_bytes, MalAttachExecMode exec_mode) {

	void* buf = (g.comm.a_rank == 0 && orig) ? orig : g_buffer_pool.acquire(total_bytes > 0 ? total_bytes : 1);
	async_broadcast_bytes(g.comm.active, buf, total_bytes, !use_async_attach_mode(exec_mode));
	return buf;

}

std::vector<char> take_pending_acc_epoch_buf(size_t fallback_size, int dtype_tag, int dop_tag) {

	if (!g.pending || g.pending->next_acc >= g.pending->acc_epoch_bufs.size()) {

		std::vector<char> buf(fallback_size, 0);

		if (fallback_size > 0) {

			write_identity(buf.data(), dtype_tag, dop_tag, (int)fallback_size);

		}

		return buf;

	}

	return std::move(g.pending->acc_epoch_bufs[g.pending->next_acc++]);

}

StagedBuffer take_pending_shared_mat() {

	if (!g.pending || g.pending->next_shared >= g.pending->shared_mats.size()) {

		return {};

	}

	StagedBuffer buf = g.pending->shared_mats[g.pending->next_shared];
	g.pending->shared_mats[g.pending->next_shared] = {};
	g.pending->next_shared++;
	return buf;

}

StagedBuffer take_pending_shared_vec() {

	if (!g.pending || g.pending->next_shared_vec >= g.pending->shared_vecs.size()) {

		return {};

	}

	StagedBuffer buf = g.pending->shared_vecs[g.pending->next_shared_vec];
	g.pending->shared_vecs[g.pending->next_shared_vec] = {};
	g.pending->next_shared_vec++;
	return buf;

}

void sync_vec_mapping_for_current_range(MalFor& f);

void load_pending_ranges_into_loop(MalFor& f) {

	if (!g.pending || g.pending->ranges.empty()) {

		return;

	}

	std::lock_guard lk(g.sync.plan_mu);

	install_loop_plan(f, g.pending->ranges);

	g.pending->ranges.clear();
	g.sync.pending_has_ranges.store(false, std::memory_order_relaxed);

	f.current = f.start;

	if (f.user_iter) {

		*f.user_iter = f.start;

	}

	f.confirmed_iter.store(f.start - 1, std::memory_order_release);

	if (f.user_limit) {

		*f.user_limit = f.end;

	}

	f.phase.store(MAL_LOOP_ATTACHING, std::memory_order_relaxed);

	sync_vec_mapping_for_current_range(f);

}

template<typename GetAcc, typename OnResult> void batched_allreduce(int n, GetAcc get_acc, OnResult on_result) {

	if (n == 0) {

		return;

	}

	struct Meta { int dt, dp, esz; };

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

void sync_vec_mapping_for_current_range(MalFor& f) {

	for (MalVec* v : f.vecs) {

		if (!v || vec_is_fully_replicated(*v)) {

			continue;

		}

		if (v->ragged) {

			const long base = (f.plan_idx < v->ragged_bases.size()) ? v->ragged_bases[f.plan_idx] : 0;
			const long new_gs = v->ragged_row_offsets[f.start] - base;

			if (v->buf_global_start != new_gs) {

				v->buf_global_start = new_gs;
				v->sync_user_ptr();

			}

			continue;

		}

		long new_global_start = f.start - (v->plan_origin_n + current_range_local_base(f));

		if (v->buf_global_start != new_global_start) {

			v->buf_global_start = new_global_start;
			v->sync_user_ptr();

		}

	}

}

void append_done_segments(MalVec& v, const MalFor& f, long local_origin, long from_local, long to_local) {

	if (MAL_UNLIKELY(to_local <= from_local)) {

		return;

	}

	if (MAL_LIKELY(f.plan_ranges.size() == 1)) {

		const auto [gs, ge] = f.plan_ranges[0];
		const long base = local_origin + f.plan_local_bases[0];
		const long rs = std::max(from_local, base);
		const long re = std::min(to_local, base + (ge - gs));

		if (rs < re) {

			v.done_segs.push_back({gs + (rs - base), re - rs});

		}

		return;

	}

	v.done_segs.reserve(v.done_segs.size() + f.plan_ranges.size());

	for (size_t ri = 0; ri < f.plan_ranges.size(); ri++) {

		const auto [gs, ge] = f.plan_ranges[ri];
		const long base = local_origin + f.plan_local_bases[ri];
		const long rs = std::max(from_local, base);
		const long re = std::min(to_local, base + (ge - gs));

		if (rs < re) {

			v.done_segs.push_back({gs + (rs - base), re - rs});

		}

	}

}

void freeze_loop_at_current(MalFor& f) {

	long cur = *f.user_iter;

	f.start = cur;
	set_limit(f, cur);
	set_iter(f, cur);

	f.plan_idx = 0;
	f.plan_ranges.clear();
	f.plan_local_bases.clear();

}

void distribute(long total, int nprocs, int rank, long& start, long& end) noexcept {

	long base = total / nprocs;
	long rem = total % nprocs;
	start = (long)rank * base + std::min((long)rank, rem);
	end = start + base + (rank < rem ? 1 : 0);

}

std::vector<long> build_partition_cuts(long total, int nprocs) {

	std::vector<long> cuts((size_t)std::max(0, nprocs) + 1, 0);
	const bool lb_enabled = g.cfg.load_balancing_enabled.load(std::memory_order_relaxed);

	if (nprocs <= 0) {

		return cuts;

	}

	if (!lb_enabled) {

		for (int r = 0; r < nprocs; r++) {

			distribute(total, nprocs, r, cuts[(size_t)r], cuts[(size_t)r + 1]);

		}

		return cuts;

	}

	std::vector<double> w;

	{

		std::lock_guard<std::mutex> lk(g.lb.weights_mu);
		w = g.lb.weights;

	}

	double sum_w = 0.0;

	if ((int)w.size() >= nprocs) {

		for (int r = 0; r < nprocs; r++) {

			sum_w += w[r];

		}

	}

	if (sum_w <= 0.0) {

		for (int r = 0; r < nprocs; r++) {

			distribute(total, nprocs, r, cuts[(size_t)r], cuts[(size_t)r + 1]);

		}

		return cuts;

	}

	std::vector<long> sizes((size_t)nprocs, 0);
	std::vector<double> rem((size_t)nprocs, 0.0);
	long base_sum = 0;

	for (int r = 0; r < nprocs; r++) {

		const double share = (w[r] / sum_w) * (double)total;
		sizes[(size_t)r] = (long)std::floor(share);
		rem[(size_t)r] = share - (double)sizes[(size_t)r];
		base_sum += sizes[(size_t)r];

	}

	long leftover = total - base_sum;

	if (leftover > 0) {

		std::vector<int> idx((size_t)nprocs);
		std::iota(idx.begin(), idx.end(), 0);
		std::sort(idx.begin(), idx.end(), [&](int a, int b) { return rem[(size_t)a] > rem[(size_t)b]; });

		for (long k = 0; k < leftover && k < (long)nprocs; k++) {

			sizes[(size_t)idx[(size_t)k]]++;

		}

	}

	if (total >= (long)nprocs) {

		for (int r = 0; r < nprocs; r++) {

			if (sizes[(size_t)r] != 0) {

				continue;

			}

			int donor = -1;
			long max_sz = 1;

			for (int j = 0; j < nprocs; j++) {

				if (sizes[(size_t)j] > max_sz) {

					max_sz = sizes[(size_t)j];
					donor = j;

				}

			}

			if (donor < 0) {

				break;

			}

			sizes[(size_t)donor]--;
			sizes[(size_t)r]++;

		}

	}

	cuts[0] = 0;

	for (int r = 0; r < nprocs; r++) {

		cuts[(size_t)r + 1] = cuts[(size_t)r] + sizes[(size_t)r];

	}

	return cuts;

}

void weighted_distribute(long total, int nprocs, int rank, long& vstart, long& vend) noexcept {

	const auto cuts = build_partition_cuts(total, nprocs);

	if (rank < 0 || rank + 1 >= (int)cuts.size()) {

		vstart = 0;
		vend = 0;
		return;

	}

	vstart = cuts[(size_t)rank];
	vend = cuts[(size_t)rank + 1];

}

std::vector<int> make_displs(const std::vector<int>& counts) {

	std::vector<int> d(counts.size());
	std::exclusive_scan(counts.begin(), counts.end(), d.begin(), 0);

	return d;

}

MalCollapseSpec mal_make_collapse_spec(const long* extents, size_t ndims) {

	MalCollapseSpec spec;

	if (!extents || ndims == 0) {

		spec.total_iters = 0;

		return spec;

	}

	spec.extents.assign(extents, extents + ndims);
	spec.strides.assign(ndims, 1);

	long total = 1;

	for (size_t i = 0; i < ndims; i++) {

		if (spec.extents[i] < 0) {

			spec.total_iters = 0;

			return spec;

		}

		if (spec.extents[i] == 0) {

			spec.total_iters = 0;

			return spec;

		}

		if (total > LONG_MAX / spec.extents[i]) {

			spec.total_iters = 0;

			return spec;

		}

		total *= spec.extents[i];

	}

	spec.total_iters = total;

	for (size_t i = ndims; i-- > 0;) {

		if (i + 1 < ndims) {

			spec.strides[i] = spec.strides[i + 1] * spec.extents[i + 1];

		}

	}

	return spec;

}

MalFor mal_for_collapse(const MalCollapseSpec& spec, long& iter, long& limit) {

	return mal_for(spec.total_iters, iter, limit);

}

void mal_collapse_decode(const MalCollapseSpec& spec, long flat_iter, long* indices_out) {

	if (!indices_out || spec.extents.empty() || spec.strides.size() != spec.extents.size()) {

		return;

	}

	if (flat_iter < 0) {

		for (size_t i = 0; i < spec.extents.size(); i++) {

			indices_out[i] = 0;

		}

		return;

	}

	for (size_t i = 0; i < spec.extents.size(); i++) {

		long extent = spec.extents[i];
		long stride = spec.strides[i];

		if (extent <= 0 || stride <= 0) {

			indices_out[i] = 0;

			continue;

		}

		indices_out[i] = (flat_iter / stride) % extent;

	}

}

MalForND mal_for_nd_begin(long* const* vars, const long* starts, const long* limits, size_t ndims) {

	MalForND out;

	if (!vars || !starts || !limits || ndims == 0) {

		return out;

	}

	out.iter_vars.assign(vars, vars + ndims);
	out.starts.assign(starts, starts + ndims);
	out.limits.assign(limits, limits + ndims);

	std::vector<long> extents(ndims, 0);

	for (size_t d = 0; d < ndims; d++) {

		extents[d] = limits[d] - starts[d];

		if (extents[d] < 0) {

			extents[d] = 0;

		}

	}

	out.spec = mal_make_collapse_spec(extents.data(), ndims);
	out.decoded_idx.assign(ndims, 0);
	out.base = std::make_unique<MalFor>(mal_for_collapse(out.spec, out.flat, out.flat_limit));

	out.done = (out.flat >= out.flat_limit);

	if (!out.done && out.iter_vars.size() == ndims) {

		mal_collapse_decode(out.spec, out.flat, out.decoded_idx.data());
		out.last_flat = out.flat;

		for (size_t d = 0; d < ndims; d++) {

			if (out.iter_vars[d]) {

				*out.iter_vars[d] = out.starts[d] + out.decoded_idx[d];

			}

		}

	}

	return out;

}

MalForND mal_for_nd_begin(long* const* iter_vars, long* const* limit_vars, const long* starts, const long* limits, size_t ndims) {

	MalForND out = mal_for_nd_begin(iter_vars, starts, limits, ndims);

	out.limit_vars.assign(limit_vars, limit_vars + ndims);

	for (size_t d = 0; d < ndims && d < out.limit_vars.size(); d++) {

		if (out.limit_vars[d]) {

			*out.limit_vars[d] = out.limits[d];

		}

	}

	return out;

}

inline bool mal_for_nd_done(const MalForND& f) {

	return f.done || !f.base || f.flat >= f.flat_limit;

}

void mal_for_nd_sync_limits(MalForND& f) {

	for (size_t d = 0; d < f.limit_vars.size() && d < f.limits.size(); d++) {

		if (f.limit_vars[d]) {

			*f.limit_vars[d] = f.limits[d];

		}

	}

}

void mal_for_nd_set_iters_from_flat(MalForND& f, long flat_iter, bool for_post_check) {

	if (f.spec.extents.empty() || flat_iter < 0) {

		return;

	}

	if (f.decoded_idx.size() != f.spec.extents.size()) {

		f.decoded_idx.assign(f.spec.extents.size(), 0);

	}

	mal_collapse_decode(f.spec, flat_iter, f.decoded_idx.data());

	const size_t ndims = f.decoded_idx.size();

	for (size_t d = 0; d < ndims && d < f.iter_vars.size() && d < f.starts.size(); d++) {

		if (!f.iter_vars[d]) {

			continue;

		}

		long v = f.starts[d] + f.decoded_idx[d];

		if (for_post_check && d + 1 == ndims) {

			v -= 1;

		}

		*f.iter_vars[d] = v;

	}

}

void mal_for_nd_mark_done(MalForND& f) {

	f.done = true;

	mal_for_nd_sync_limits(f);

	for (size_t d = 0; d < f.iter_vars.size() && d < f.limits.size(); d++) {

		if (f.iter_vars[d]) {

			*f.iter_vars[d] = f.limits[d];

		}

	}

}

void mal_check_for(MalForND& f) {

	if (mal_for_nd_done(f)) {

		return;

	}

	mal_check_for(*f.base);

	long scheduled_flat = f.flat;

	if (scheduled_flat >= f.flat_limit) {

		mal_for_nd_mark_done(f);

		return;

	}

	long next_flat = scheduled_flat + 1;

	if (next_flat >= f.flat_limit) {

		mal_for_nd_mark_done(f);

		return;

	}

	if (scheduled_flat == f.last_flat && !f.decoded_idx.empty() && f.decoded_idx.size() == f.spec.extents.size()) {

		long& inner = f.decoded_idx.back();

		if (inner + 1 < f.spec.extents.back()) {

			inner++;
			f.flat = next_flat;
			f.last_flat = next_flat;

			const size_t d = f.decoded_idx.size() - 1;

			if (d < f.iter_vars.size() && d < f.starts.size() && f.iter_vars[d]) {

				*f.iter_vars[d] = f.starts[d] + inner - 1;

			}

			return;

		}

	}

	f.flat = next_flat;
	f.last_flat = next_flat;
	mal_for_nd_sync_limits(f);
	mal_for_nd_set_iters_from_flat(f, next_flat, true);

}

MalFor& mal_for_nd_base(MalForND& f) {

	if (MAL_UNLIKELY(!f.base)) {

		MAL_LOG(MAL_LOG_ERROR, "mal_for_nd_base called on uninitialized MalForND (base is null)");
		std::abort();

	}

	return *f.base;

}

void MalVec::free_resources() {

	if (buf) {

		g_buffer_pool.release(buf, buf_bytes > 0 ? buf_bytes : 1);
		buf = nullptr;
		buf_bytes = 0;

	}

	if (ragged_full_src) {

		std::free(ragged_full_src);
		ragged_full_src = nullptr;
		ragged_full_bytes = 0;

	}

}

void SharedMat::free_resources() {

	if (!user_owned && buf) {

		g_buffer_pool.release(buf, total_bytes > 0 ? total_bytes : 1);

	}

	buf = nullptr;

	if (user_ptr) {

		*user_ptr = nullptr;

	}

}

std::vector<std::pair<long,long>> slice_remaining(const std::vector<std::pair<long,long>>& remaining, std::vector<long>& offsets, long vstart, long vend) {

	std::vector<std::pair<long,long>> out;
	out.reserve(remaining.size());

	if (remaining.empty() || vend <= vstart || offsets.empty()) {

		return out;

	}

	auto ub = std::upper_bound(offsets.begin(), offsets.end(), vstart);
	size_t ub_dist = (size_t)std::distance(offsets.begin(), ub);
	size_t idx = (ub_dist == 0) ? 0 : ub_dist - 1;
	long offset = offsets[idx];

	for (; idx < remaining.size(); idx++) {

		auto [a, b] = remaining[idx];

		long len = b - a;

		if (MAL_UNLIKELY(offset + len <= vstart)) {

			offset += len;

			continue;

		}

		if (MAL_UNLIKELY(offset >= vend)) {

			break;

		}

		long s = a + std::max(0L, vstart - offset);
		long e = a + std::min(len, vend - offset);

		if (s < e) {

			out.push_back({s, e});

		}

		offset += len;

	}

	return out;

}

bool vec_can_reuse_assigned_ranges(const MalVec& v, const std::vector<std::pair<long,long>>& assigned) {

	if (v.access_mode != MAL_ACCESS_READ_ONLY || !v.cache_valid) {

		return false;

	}

	if (assigned.empty()) {

		return true;

	}

	for (const auto& rg : assigned) {

		const long s = rg.first;
		const long e = rg.second;

		if (s < v.cache_start || e > v.cache_end || s > e) {

			return false;

		}

	}

	return true;

}

bool vec_reuse_local_copy(MalVec& v, const std::vector<std::pair<long,long>>& assigned, long done_n) {

	if (!vec_can_reuse_assigned_ranges(v, assigned)) {

		return false;

	}

	if (assigned.empty()) {

		return true;

	}

	if (done_n < 0) {

		return false;

	}

	long dst_off = done_n;

	for (const auto& rg : assigned) {

		const long s = rg.first;
		const long e = rg.second;
		const long len = e - s;

		if (len <= 0) {

			continue;

		}

		const long src_off = v.cache_local_off + (s - v.cache_start);

		if (src_off < 0 || src_off + len > v.local_n) {

			return false;

		}

		if (dst_off != src_off) {

			std::memmove(static_cast<char*>(v.buf) + dst_off * (long)v.elem_size, static_cast<char*>(v.buf) + src_off * (long)v.elem_size, (size_t)len * v.elem_size);

		}

		dst_off += len;

	}

	return true;

}

#endif
