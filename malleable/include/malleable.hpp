/* malleable - Malleable MPI runtime: loops whose set of active ranks can change while running

    FEATURES:
        - Malleable loops: mal_for() + mal_check_for() split an index range among the active ranks
        - Runtime resize: the active rank set grows, shrinks or rebalances between epochs, no restart
        - Attached data (vectors, matrices, CSR, accumulators) migrates with its iterations on resize
        - Results gathered or reduced into one rank on mal_finalize()
        - Pluggable resize decisions: built-in policies, a function pointer or a dlopen()'d plugin
        - Distributed decision: every active rank votes, a quorum + median consensus picks the size
        - Throughput-weighted load balancing of iteration ranges
        - Nested loops collapsed into one malleable loop: mal_for_nd_begin()
        - Iterative (stencil-like) kernels: mal_loop_horizon() + mal_step()
        - Runtime worker thread pinning and self-accounting (CPU time, run queue delay)
        - Leveled logging and CSV tracing to stdout

    NOTES:
        - Built on MPI Sessions: mal_init() creates the session, do NOT call MPI_Init()/MPI_Finalize()
        - Every rank runs the same program: inactive ranks block inside mal_for()/mal_check_for()
          until the runtime hands them work or the loop ends
        - Malleability is OFF by default (MAL_MALLEABILITY_ENABLED=0): all ranks run the whole loop
          and no resize is ever evaluated
        - mal_init() starts one runtime worker thread per rank; resize decision functions run on
          that thread, concurrently with the loop body, share state with it through atomics only
        - Register resize decision functions BEFORE mal_init(), a registered one always takes
          precedence over the built-in policy selected by mal_init() or MAL_RESIZE_POLICY
        - One malleable loop at a time: a new mal_for() seals the data attached to the previous one,
          keep the MalFor/MalForND alive until the loop ends
        - Attached pointers are rebased to global indices: iteration i always accesses (*user_ptr)[i]
        - mal_rank()/mal_size() refer to the universe (every launched rank), mal_active_size() to
          the active set

    USAGE:
        mal_init();
        double *data = (mal_rank() == 0)? load_data(N) : nullptr;
        long i, lim;
        MalFor f = mal_for(N, i, lim);
        mal_attach_vec(f, (void**)&data, sizeof(double), N, 0); // Gather result into rank 0
        for (; i < lim; i++) {
            data[i] = compute(i);
            mal_check_for(f); // Last call of every iteration
        }
        mal_finalize(); // data is complete on rank 0

    DEPENDENCIES:
        MPI 4.0 or later: MPI Sessions with MPI_THREAD_MULTIPLE
        dlopen()/dlsym(): resize decision plugins
        Linux: thread pinning and worker accounting (partial support on macOS)

    CONFIGURATION:
        #define MAL_LOG_DISABLED
            Compile out every MAL_LOG()/MAL_LOG_L() call in the including translation unit

    ENVIRONMENT VARIABLES (read on mal_init()):
        MAL_MALLEABILITY_ENABLED=0      Master switch: allow the active rank set to change at runtime
        MAL_RESIZE_ENABLED=1            Allow size-changing resizes (see mal_get_resize_enabled())
        MAL_LOAD_BALANCING_ENABLED=0    Throughput-weighted iteration ranges and same-size rebalances
        MAL_INITIAL_SIZE=<universe>     Number of active ranks at start
        MAL_RESIZE_POLICY               Built-in policy: auto, throughput, efficiency|energy,
                                        fixed|fixed_sequence, cost (overrides mal_init() argument)
        MAL_RESIZE_SEQ                  fixed policy: comma-separated active sizes to visit, in order
        MAL_RESIZE_QUORUM=1.0           Fraction of voting ranks that must agree on a direction (0.5, 1]
        MAL_EPOCH_INTERVAL_MS=1000      Time between resize evaluations
        MAL_FAST_RESPONSE=0             No cooldown epochs after a resize
        MAL_STENCIL_EPOCH_STEPS=64      mal_step(): evaluate resize every N steps
        MAL_STENCIL_RESID_REDUCES=1     mal_step(): residual allreduces run on non-evaluation steps
        MAL_EPOCH_CHANGE_MODE=1         Parsed, currently unused
        MAL_AFFINITY=1                  Pin main thread to a P-core and worker thread to an E-core
        MAL_MAIN_CORE=<auto>            Core for the main thread
        MAL_WORKER_CORE=<auto>          Core for the runtime worker thread
        MAL_LOG_LEVEL=INFO              DEBUG, INFO, WARN, ERROR, NONE or 0..4
        MAL_LOG_ALL_RANKS=0             Log from every rank, not only rank 0
        MAL_TRACE_ENABLED=0             Print CSV trace rows (START, END, RESIZE, TIMING...) to stdout
        MAL_TIMING=0                    Collect internal timers, printed as TIMING rows on mal_finalize()
                                        (requires MAL_TRACE_ENABLED=1)
        MAL_BASELINE_FROM_PERRANK=0     auto policies: estimate 1-rank throughput as global_thr/active_n
        MAL_COST_KEEP_FRACTION=0.97     cost policy: fewest ranks keeping this fraction of peak throughput
        MAL_COST_SAMPLE_STEP=8          cost policy, iterative kernels: rank step between sampled sizes
        MAL_COST_SAMPLE_REFINE=0        cost policy, iterative kernels: refine around best size with step/4
        MAL_COST_SAMPLE_MEAS=3          cost policy, iterative kernels: epochs measured per sampled size


    LICENSE: MIT

    Copyright (c) 2026 Pablo Liste Cancela

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

*/

#pragma once

#include <mpi.h>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

// Function specifier: symbols stay visible when the library is built with -fvisibility=hidden
#define MAL_API __attribute__((visibility("default")))

// Throughput (iterations/s) below this value is considered zero
constexpr double kEpsThroughput = 1e-9;

// Log a printf-style message on this rank if level passes mal_should_log()
// Output: [t=<seconds since mal_init()>][<LEVEL>][R<rank>] | <message>
// NOTE: Compiled out when MAL_LOG_DISABLED is defined
#if !defined(MAL_LOG_DISABLED)
#define MAL_LOG(level, fmt, ...)                                    \
	do {                                                        \
		MalLogLevel _mal_level = (level);                   \
		if (mal_should_log(_mal_level))                     \
			printf("[t=%10.4f][%-5s][R%d] | " fmt "\n", \
				MPI_Wtime() - mal_t_origin(),       \
				mal_log_level_name(_mal_level),     \
				mal_rank(),                         \
				##__VA_ARGS__);                     \
	} while (0)

// Log a printf-style message with a subsystem tag, same as MAL_LOG()
// Output: [t=<seconds since mal_init()>][<LEVEL>][<tag>][R<rank>] | <message>
#define MAL_LOG_L(level, tag, fmt, ...)                                   \
	do {                                                              \
		MalLogLevel _mal_level = (level);                         \
		if (mal_should_log(_mal_level))                           \
			printf("[t=%10.4f][%-5s][%-6s][R%d] | " fmt "\n", \
				MPI_Wtime() - mal_t_origin(),             \
				mal_log_level_name(_mal_level),           \
				(tag),                                    \
				mal_rank(),                               \
				##__VA_ARGS__);                           \
	} while (0)
#else
#define MAL_LOG(level, fmt, ...)         ((void)0)
#define MAL_LOG_L(level, tag, fmt, ...)  ((void)0)
#endif

// Runtime-internal attached data, opaque to users
struct MalVec;
struct MalAcc;

// Log level
// NOTE: Organized by priority level, messages below MAL_LOG_LEVEL are discarded
enum MalLogLevel {

	MAL_LOG_DEBUG,  // Debug logging: epochs, votes, transfers, configuration
	MAL_LOG_INFO,   // Info logging: program progress and results (default)
	MAL_LOG_WARN,   // Warning logging: recoverable failures, ignored configuration
	MAL_LOG_ERROR,  // Error logging: unrecoverable failures, usually followed by an abort
	MAL_LOG_NONE,   // Disable logging
};

// Malleable loop phase, as seen by this rank
enum MalLoopPhase {

	MAL_LOOP_WAITING_ACTIVATION,    // Rank has no iterations assigned, waiting for work
	MAL_LOOP_ATTACHING,             // Attached data is being distributed
	MAL_LOOP_RUNNING,               // Executing iterations
	MAL_LOOP_FINISHED,              // Loop ended: finalized, destroyed or moved-from
};

// Attached data distribution policy
enum MalAttachPolicy {

	MAL_ATTACH_PARTITIONED,     // Split in blocks following the iteration ranges, migrated on resize
	MAL_ATTACH_SHARED_ACTIVE,   // Full copy on every active rank, freed on leaving ranks, resent to joining ones
	MAL_ATTACH_SHARED_ALL,      // Full copy kept on leaving ranks, read-only data is not resent when they rejoin
};

// Attached data transfer execution mode
enum MalAttachExecMode {

	MAL_ATTACH_INHERIT, // Use the runtime default (MAL_ATTACH_SYNC)
	MAL_ATTACH_SYNC,    // Attach call blocks until the data is distributed
	MAL_ATTACH_ASYNC,   // Transfer runs on the worker thread, mal_check_for() waits for it
};

// Attached data access mode
enum MalDataAccessMode {

	MAL_ACCESS_READ_WRITE,  // Loop writes the data: completed elements are tracked, migrated and gathered
	MAL_ACCESS_READ_ONLY,   // Loop only reads the data: copies are cached and reused instead of resent
};

// Resize policy, selects the decision function installed by mal_init()
// NOTE: MAL_RESIZE_POLICY environment variable overrides it
enum MalResizePolicy {

	MAL_RESIZE_POLICY_AUTO,             // Search the largest size with parallel efficiency >= 0.6
	MAL_RESIZE_POLICY_THROUGHPUT,       // Search the largest size with parallel efficiency >= 0.0 (max throughput)
	MAL_RESIZE_POLICY_EFFICIENCY,       // Search the largest size with parallel efficiency >= 0.8
	MAL_RESIZE_POLICY_FIXED_SEQUENCE,   // Visit the sizes listed in MAL_RESIZE_SEQ, one per resize
	MAL_RESIZE_POLICY_COST,             // Find peak throughput, then settle on fewest ranks keeping MAL_COST_KEEP_FRACTION of it
	MAL_RESIZE_POLICY_CUSTOM,           // User decision function (falls back to AUTO if none is registered)
};

// Resize vote of one rank
enum MalVote {

	MAL_VOTE_KEEP,      // Stay at the current active size
	MAL_VOTE_RESIZE,    // Move to ResizeDecision.target_active_size
	MAL_VOTE_ABSTAIN,   // Do not take part in this epoch's consensus
};

// ResizeDecision, returned by a decision function once per epoch on every active rank
// NOTE: Votes are combined: a unanimous vote is applied directly; otherwise at least half of the active
// ranks must not abstain, the winning direction (grow, shrink or same-size rebalance) needs a majority
// and MAL_RESIZE_QUORUM of the votes, and its median target (the one closest to current size) is used
struct ResizeDecision {

	MalVote vote{MAL_VOTE_KEEP}; // Rank vote: keep, resize or abstain
	bool done{false}; // Loop is finished: stop evaluating resizes for it on all ranks
	int target_active_size{-1}; // Desired active size when voting MAL_VOTE_RESIZE, 1..mal_size() (current size: rebalance)
	bool settled{false}; // Policy converged: mal_step() kernels stop per-epoch evaluation
};

// EpochMetrics, reduced over all ranks and passed to the decision function every epoch
// NOTE: Throughputs in iterations/s, times in seconds
struct EpochMetrics {

	double global_thr{0.0}; // Sum of the active ranks' throughput in this epoch
	double global_remaining{0.0}; // Iterations left on all ranks (includes steps declared with mal_loop_horizon())
	int active_n{0}; // Current number of active ranks
	bool any_has_loop{false}; // At least one rank is running a malleable loop
	double max_thr{0.0}; // Throughput of the fastest active rank
	double min_thr{0.0}; // Throughput of the slowest active rank (0.0 if none measured)
	double max_rem_time{0.0}; // Largest estimated remaining time of a rank
	double sum_rem_time{0.0}; // Sum of estimated remaining times of all ranks
	int max_slow_streak{0}; // Longest run of consecutive epochs a rank stayed > 1.5x the mean remaining time
	bool any_settled{false}; // At least one rank's last decision was settled
	int resize_commit_count{0}; // Resizes and rebalances committed so far
	int resize_cooldown_remaining{0}; // Epochs left of cooldown after last resize (advisory, not enforced)
	int rebalance_cooldown_remaining{0}; // Epochs left of cooldown after last rebalance (advisory, not enforced)
	double epoch_elapsed{0.0}; // Time elapsed in the current epoch (max over active ranks)
	int epoch_interval_ms{0}; // Configured epoch interval (MAL_EPOCH_INTERVAL_MS)
	bool iterative_kernel{false}; // Loop is an iterative kernel (mal_loop_horizon() was called)

	// Get load imbalance: max_rem_time over mean remaining time (1.0: balanced, 0.0: unknown)
	double imbalance_ratio() const {

		if (active_n <= 0 || sum_rem_time <= kEpsThroughput) {

			return 0.0;
		}

		const double avg = sum_rem_time / (double)active_n;
		return (avg > kEpsThroughput) ? max_rem_time / avg : 0.0;
	}
};

// ResizeStateBlob, serialized decision function state
struct ResizeStateBlob {

	const void* data{nullptr}; // State data, copied by the runtime right after the save callback returns
	size_t len{0}; // State data size in bytes
};

// Resize decision function: called once per epoch on every active rank, from the runtime worker thread
using DecideResizeFunc = ResizeDecision (*)(const EpochMetrics& m);

// Decision state save callback: called on active rank 0 after every committed resize
using ResizeStateSaveFunc = ResizeStateBlob (*)();

// Decision state load callback: called on every active rank with rank 0's state after every committed resize
using ResizeStateLoadFunc = void (*)(const void* data, size_t len);

// MpiType, MPI datatype of a basic C++ type, used by mal_attach_acc(), mal_sync() and mal_bcast()
template<typename T> struct MpiType;
template<> struct MpiType<int> {
	static MPI_Datatype value() { return MPI_INT; }
};
template<> struct MpiType<long> {
	static MPI_Datatype value() { return MPI_LONG; }
};
template<> struct MpiType<long long> {
	static MPI_Datatype value() { return MPI_LONG_LONG; }
};
template<> struct MpiType<unsigned> {
	static MPI_Datatype value() { return MPI_UNSIGNED; }
};
template<> struct MpiType<unsigned long> {
	static MPI_Datatype value() { return MPI_UNSIGNED_LONG; }
};
template<> struct MpiType<float> {
	static MPI_Datatype value() { return MPI_FLOAT; }
};
template<> struct MpiType<double> {
	static MPI_Datatype value() { return MPI_DOUBLE; }
};

// MalFor, malleable loop state returned by mal_for()
// NOTE: Iteration ranges are global [start, end) indices, move-only
struct MAL_API alignas(64) MalFor {

	long start{0}; // First iteration of the current range
	long end{0}; // One past the last iteration of the current range
	long current{0}; // Last iteration seen by mal_check_for()
	long* user_iter{nullptr}; // User iteration variable
	long* user_limit{nullptr}; // User limit variable
	std::atomic<MalLoopPhase> phase{MAL_LOOP_WAITING_ACTIVATION}; // Loop phase on this rank
	std::atomic<long> confirmed_iter{LONG_MIN}; // Last completed iteration, read by the worker thread
	size_t plan_idx{0}; // Index of the current range in plan_ranges
	size_t check_counter{0}; // Number of mal_check_for() calls
	unsigned long long gen{0}; // Loop generation, increased by every mal_for()

	std::vector<std::pair<long, long>> plan_ranges; // Iteration ranges assigned to this rank
	std::vector<long> plan_local_bases; // Local buffer offset of every range in plan_ranges
	std::vector<MalVec*> vecs; // Vectors attached to this loop (partitioned matrices included)
	std::vector<MalAcc*> accs; // Accumulators attached to this loop

	MalFor() = default;
	~MalFor();
	MalFor(const MalFor&) = delete;
	MalFor& operator=(const MalFor&) = delete;
	MalFor(MalFor&& other) noexcept;
	MalFor& operator=(MalFor&&) = delete;
};

// MalCollapseSpec, shape of a nested loop collapsed into one flat loop (row-major)
struct MalCollapseSpec {

	std::vector<long> extents; // Iteration count of every dimension
	std::vector<long> strides; // Flat iteration stride of every dimension
	long total_iters{0}; // Product of extents: flat loop size
};

// MalForND, collapsed nested loop state returned by mal_for_nd_begin()
struct MalForND {

	std::unique_ptr<MalFor> base; // Underlying flat malleable loop
	MalCollapseSpec spec; // Nested loop shape
	std::vector<long*> iter_vars; // User iteration variable of every dimension
	std::vector<long*> limit_vars; // User limit variable of every dimension (empty if not requested)
	std::vector<long> starts; // First index of every dimension
	std::vector<long> limits; // End index (exclusive) of every dimension
	std::vector<long> decoded_idx; // Current zero-based index of every dimension
	long flat{0}; // Flat iteration, base loop iteration variable
	long flat_limit{0}; // Flat limit, base loop limit variable
	long last_flat{LONG_MIN}; // Last decoded flat iteration
	bool done{true}; // Nested loop finished on this rank

	MalForND() = default;
	MalForND(const MalForND&) = delete;
	MalForND& operator=(const MalForND&) = delete;

	MalForND(MalForND&& other) noexcept {

		*this = std::move(other);
	}

	MalForND& operator=(MalForND&& other) noexcept {

		if (this == &other) {

			return *this;
		}

		base = std::move(other.base);
		spec = std::move(other.spec);
		iter_vars = std::move(other.iter_vars);
		limit_vars = std::move(other.limit_vars);
		starts = std::move(other.starts);
		limits = std::move(other.limits);
		decoded_idx = std::move(other.decoded_idx);
		flat = other.flat;
		flat_limit = other.flat_limit;
		last_flat = other.last_flat;
		done = other.done;

		if (base) {

			base->user_iter = &flat;
			base->user_limit = &flat_limit;
		}

		return *this;
	}
};

// Runtime lifecycle functions
MAL_API void mal_init(MalResizePolicy policy = MAL_RESIZE_POLICY_CUSTOM); // Initialize MPI session, active set and worker thread, read environment configuration
MAL_API void mal_finalize(); // Stop worker, gather/reduce attached results into their result ranks, finalize MPI session

// Universe and active set functions
MAL_API int mal_rank(); // Get rank of this process in the universe
MAL_API int mal_size(); // Get universe size: every launched rank, maximum active size
MAL_API int mal_active_size(); // Get current number of active ranks (0 on inactive ranks)
MAL_API double mal_t_origin(); // Get MPI_Wtime() at mal_init(), time origin of logs and traces

// Runtime worker thread functions
MAL_API long mal_worker_tid(); // Get Linux thread id of the worker thread (-1 if unknown)
MAL_API int mal_worker_core(); // Get core the worker thread is pinned to, else last core it ran on (-1 if unknown)
MAL_API double mal_worker_cpu_seconds(); // Get CPU time used by the worker thread in seconds (-1.0 if unavailable)
MAL_API double mal_worker_runq_seconds(); // Get time the worker thread waited in the run queue in seconds, sampled every epoch (-1.0 if unavailable)

// Logging functions
MAL_API const char* mal_log_level_name(MalLogLevel level); // Get log level name: "DEBUG", "INFO", "WARN", "ERROR" or "NONE"
MAL_API bool mal_should_log(MalLogLevel level); // Check if level passes MAL_LOG_LEVEL on this rank (rank 0 only unless MAL_LOG_ALL_RANKS)

// Runtime configuration functions
MAL_API void mal_set_resize_enabled(bool enabled); // Enable/disable size-changing resizes at runtime (MAL_RESIZE_ENABLED)
[[nodiscard]] MAL_API bool mal_get_resize_enabled(); // Check if size-changing resizes are enabled, decision functions should honor it
[[nodiscard]] MAL_API bool mal_get_load_balancing_enabled(); // Check if load balancing is enabled (MAL_LOAD_BALANCING_ENABLED)
MAL_API void mal_set_resize_min_horizon_epochs(int epochs); // Set remaining-time horizon, in epochs, below which ranks vote KEEP without calling the decision function (default: 2, 0: disabled)

// Environment variable parsing functions
// NOTE: Invalid values are ignored with a warning and fallback is returned
[[nodiscard]] MAL_API bool mal_env_bool(const char* name, bool fallback); // Get boolean environment variable: 0/1, true/false, on/off, yes/no
[[nodiscard]] MAL_API long mal_env_long(const char* name, long fallback, long min, long max); // Get integer environment variable in [min, max]
[[nodiscard]] MAL_API double mal_env_double(const char* name, double fallback, double min, double max); // Get floating-point environment variable in [min, max]

// Resize decision functions
// NOTE: Call them before mal_init()
MAL_API void mal_set_decide_resize_func(DecideResizeFunc func); // Set resize decision function
MAL_API void mal_set_decide_resize_plugin(const char* path, const char* func_name); // Set resize decision function from a shared library (extern "C" symbol), abort on failure
MAL_API void mal_set_decide_resize_state_funcs(ResizeStateSaveFunc save, ResizeStateLoadFunc load); // Set decision state callbacks, sync state from active rank 0 after every resize
MAL_API void mal_set_decide_resize_state_plugin(const char* save_func_name, const char* load_func_name); // Set decision state callbacks from the loaded decision plugin, abort on failure

// Malleable loop functions
[[nodiscard]] MAL_API MalFor mal_for(long total_iters, long& iter, long& limit); // Begin loop over [0, total_iters): set iter/limit to this rank's range, inactive ranks wait for work
MAL_API void mal_check_for(MalFor& f); // Loop checkpoint, call last in every iteration: may move iter/limit to new work after a resize
[[nodiscard]] MAL_API MalForND mal_for_nd_begin(long* const* vars, const long* starts, const long* limits, size_t ndims); // Begin nested loop over [starts[d], limits[d]) collapsed into one loop, indices written to *vars[d]
[[nodiscard]] MAL_API MalForND mal_for_nd_begin(long* const* iter_vars, long* const* limit_vars, const long* starts, const long* limits, size_t ndims); // Begin collapsed nested loop, also write limits[d] to *limit_vars[d]
MAL_API MalFor& mal_for_nd_base(MalForND& f); // Get underlying flat loop of a nested loop (abort if not initialized)
MAL_API void mal_check_for(MalForND& f); // Nested loop checkpoint, call last in every innermost iteration
MAL_API void mal_loop_horizon(long steps_remaining); // Declare remaining steps of an iterative kernel, call before every step's mal_for()
MAL_API void mal_step(MalFor& f, void* full_buf, size_t elem_size, long total_n); // End an iterative kernel step: halo exchange of full_buf (full gather on last step), resize evaluation every MAL_STENCIL_EPOCH_STEPS

// Data attachment functions
// NOTE: Source data is read from rank 0 (*user_ptr may be nullptr on the other ranks),
// result_rank >= 0 receives the complete data in *user_ptr on mal_finalize() (-1: not gathered)
MAL_API void mal_attach_vec(MalFor& f, void** user_ptr, size_t elem_size, long total_n, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE); // Attach vector of total_n elements, element i belongs to iteration i
MAL_API void mal_attach_vec(MalForND& f, void** user_ptr, size_t elem_size, long total_n, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE); // Attach vector to a nested loop, element i belongs to flat iteration i
MAL_API void mal_attach_mat(MalFor& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE); // Attach primary_n x secondary_n row-major matrix, row i belongs to iteration i (result_rank ignored if shared)
MAL_API void mal_attach_mat(MalForND& f, void** user_ptr, size_t elem_size, long primary_n, long secondary_n, int result_rank = -1, MalAttachPolicy policy = MAL_ATTACH_PARTITIONED, MalAttachExecMode exec_mode = MAL_ATTACH_INHERIT, MalDataAccessMode access_mode = MAL_ACCESS_READ_WRITE); // Attach row-major matrix to a nested loop, row i belongs to flat iteration i
MAL_API void mal_attach_csr(MalFor& f, void** values, size_t value_elem_size, void** col_indices, size_t index_elem_size, long* row_ptr, long n_rows, long nnz); // Attach CSR matrix split by rows, row_ptr broadcast from rank 0 (requires every rank active)

// Collective functions
MAL_API void mal_bcast_impl(void* buf, int count, MPI_Datatype dtype, int root); // Broadcast count elements from active rank root to the active ranks, untyped mal_bcast()
MAL_API void mal_sync_impl(MalFor& f, void* buf, int count, MPI_Datatype dtype, MPI_Op op); // Reduce count elements in place across the active ranks, untyped mal_sync()

namespace detail {

// AccDesc, type-erased accumulator description
struct AccDesc {

	void* ptr; // User accumulator variable
	MPI_Datatype dtype; // MPI datatype of the accumulator
	MPI_Op dop; // MPI reduction operation
	size_t esz; // Accumulator size in bytes
	void (*fn_get)(const void* p, void* dst); // Copy accumulator value to dst
	void (*fn_set)(void* p, const void* src); // Set accumulator value from src
	void (*fn_add)(void* p, const void* src); // Add src to accumulator value
	void (*fn_reset)(void* p); // Reset accumulator to its default value
};

// Register an accumulator in a loop, use mal_attach_acc()
MAL_API void acc_register(MalFor& f, AccDesc d, int result_rank);

// Copy accumulator of type T
template<typename T> inline void acc_get_t(const void* p, void* d) {

	*static_cast<T*>(d) = *static_cast<const T*>(p);
}

// Set accumulator of type T
template<typename T> inline void acc_set_t(void* p, const void* s) {

	*static_cast<T*>(p) = *static_cast<const T*>(s);
}

// Add to accumulator of type T
template<typename T> inline void acc_add_t(void* p, const void* s) {

	*static_cast<T*>(p) += *static_cast<const T*>(s);
}

// Reset accumulator of type T
template<typename T> inline void acc_reset_t(void* p) {

	*static_cast<T*>(p) = T{};
}

}

// Attach accumulator reduced with op across ranks, acc is set to op's identity and holds the result on result_rank after mal_finalize()
template<typename T> inline void mal_attach_acc(MalFor& f, T& acc, MPI_Datatype dtype, MPI_Op op, int result_rank = 0) {

	detail::acc_register(f, {

								&acc,
								dtype,
								op,
								sizeof(T),
								detail::acc_get_t<T>,
								detail::acc_set_t<T>,
								detail::acc_add_t<T>,
								detail::acc_reset_t<T>,
							},
		result_rank);
}

// Attach accumulator of a basic type (MpiType), reduced with MPI_SUM
template<typename T> inline void mal_attach_acc(MalFor& f, T& acc, int result_rank = 0) {

	mal_attach_acc(f, acc, MpiType<T>::value(), MPI_SUM, result_rank);
}

// Attach accumulator of a basic type to a nested loop, reduced with MPI_SUM
template<typename T> inline void mal_attach_acc(MalForND& f, T& acc, int result_rank = 0) {

	mal_attach_acc(mal_for_nd_base(f), acc, result_rank);
}

// Reduce value across the active ranks
// NOTE: On an attached accumulator it reduces over every rank now and seals it, skipping mal_finalize()
template<typename T> inline void mal_sync(MalFor& f, T& value, MPI_Op op = MPI_SUM) {

	mal_sync_impl(f, &value, 1, MpiType<T>::value(), op);
}

// Reduce count values in place across the active ranks
template<typename T> inline void mal_sync(MalFor& f, T* values, int count, MPI_Op op = MPI_SUM) {

	mal_sync_impl(f, values, count, MpiType<T>::value(), op);
}

// Reduce value across the active ranks of a nested loop
template<typename T> inline void mal_sync(MalForND& f, T& value, MPI_Op op = MPI_SUM) {

	mal_sync_impl(mal_for_nd_base(f), &value, 1, MpiType<T>::value(), op);
}

// Broadcast value from active rank root to the active ranks
template<typename T> inline void mal_bcast(T& value, int root = 0) {

	mal_bcast_impl(&value, 1, MpiType<T>::value(), root);
}

// Broadcast count values from active rank root to the active ranks
template<typename T> inline void mal_bcast(T* values, int count, int root = 0) {

	mal_bcast_impl(values, count, MpiType<T>::value(), root);
}
