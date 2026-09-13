#ifndef THREAD_METRICS_HPP
#define THREAD_METRICS_HPP

#ifdef __linux__

	#ifndef _GNU_SOURCE

		#define _GNU_SOURCE

	#endif

#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#if defined(__linux__)

	#include <asm/unistd.h>
	#include <dirent.h>
	#include <linux/perf_event.h>
	#include <sched.h>
	#include <string.h>
	#include <sys/resource.h>
	#include <sys/syscall.h>
	#include <unistd.h>

#endif

struct TidInfo {

	long tid;
	std::string comm;
	double cpu_s;
	double runq_s;
	int last_cpu;
	int pinned_cpu;
};

struct ThreadSnap {

	double compute_cpu_s;
	double proc_cpu_s;
	double runq_s;
	long nvcsw;
	long nivcsw;
	long long cycles;
	long long insns;
	long long llc_ref;
	long long llc_miss;
	long long ref_cycles;
};

struct PerfCounters {

	int fd_cycles;
	int fd_insns;
	int fd_llc_ref;
	int fd_llc_miss;
	int fd_ref_cyc;
};

static inline double tm_clock_s(clockid_t id) {

	struct timespec ts;

	if (clock_gettime(id, &ts) != 0) {

		return -1.0;
	}

	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double tm_mono_s() {

	return tm_clock_s(CLOCK_MONOTONIC);
}

static inline double tm_thread_cpu_s() {

	return tm_clock_s(CLOCK_THREAD_CPUTIME_ID);
}

static inline double tm_proc_cpu_s() {

	return tm_clock_s(CLOCK_PROCESS_CPUTIME_ID);
}

static inline long tm_self_tid() {

#if defined(__linux__)

	return (long)syscall(SYS_gettid);

#else

	return -1;

#endif
}

static inline int tm_current_cpu() {

#if defined(__linux__)

	return sched_getcpu();

#else

	return -1;

#endif
}

static inline double tm_read_runq_s(const char* path) {

#if defined(__linux__)

	FILE* f = std::fopen(path, "r");

	if (!f) {

		return -1.0;
	}

	unsigned long long on_cpu = 0, run_delay = 0;
	unsigned long slices = 0;
	const int n = std::fscanf(f, "%llu %llu %lu", &on_cpu, &run_delay, &slices);
	std::fclose(f);

	if (n < 2) {

		return -1.0;
	}

	return (double)run_delay * 1e-9;

#else

	(void)path;

	return -1.0;

#endif
}

static inline double tm_thread_runq_s() {

	return tm_read_runq_s("/proc/thread-self/schedstat");
}

static inline bool tm_schedstats_enabled() {

#if defined(__linux__)

	double a = tm_read_runq_s("/proc/thread-self/schedstat");

	if (a < 0.0) {

		return false;
	}

	FILE* f = std::fopen("/proc/sys/kernel/sched_schedstats", "r");

	if (f) {

		int v = 0;
		const int n = std::fscanf(f, "%d", &v);
		std::fclose(f);

		if (n == 1 && v == 1) {

			return true;
		}
	}

	volatile double spin = 0.0;

	for (long k = 0; k < 200000; k++) {

		spin += (double)k;
	}

	(void)spin;

	double b = tm_read_runq_s("/proc/thread-self/schedstat");

	return (b > a);

#else

	return false;

#endif
}

static inline void tm_rusage_thread(long& nvcsw, long& nivcsw) {

#if defined(__linux__) && defined(RUSAGE_THREAD)

	struct rusage ru;

	if (getrusage(RUSAGE_THREAD, &ru) == 0) {

		nvcsw = ru.ru_nvcsw;
		nivcsw = ru.ru_nivcsw;

		return;
	}

#endif

	nvcsw = -1;
	nivcsw = -1;
}

#if defined(__linux__)

static inline long tm_gettid() {

	return (long)syscall(SYS_gettid);
}

static inline int tm_perf_open_one(unsigned int type, unsigned long long config) {

	struct perf_event_attr attr;
	std::memset(&attr, 0, sizeof(attr));
	attr.type = type;
	attr.size = sizeof(attr);
	attr.config = config;
	attr.disabled = 0;
	attr.exclude_kernel = 1;
	attr.exclude_hv = 1;
	attr.inherit = 0;
	attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

	return (int)syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
}

static inline bool tm_parse_stat(const char* path, double& cpu_s, int& last_cpu) {

	FILE* f = std::fopen(path, "r");

	if (!f) {

		return false;
	}

	char buf[4096];
	const size_t got = std::fread(buf, 1, sizeof(buf) - 1, f);
	std::fclose(f);

	if (got == 0) {

		return false;
	}

	buf[got] = '\0';

	char* rp = std::strrchr(buf, ')');

	if (!rp) {

		return false;
	}

	std::vector<std::string> tok;
	char* save = nullptr;
	char* p = ::strtok_r(rp + 1, " \t\n", &save);

	while (p) {

		tok.push_back(p);
		p = ::strtok_r(nullptr, " \t\n", &save);
	}

	if (tok.size() < 37) {

		return false;
	}

	const double hz = (double)sysconf(_SC_CLK_TCK);
	const double ut = std::strtod(tok[11].c_str(), nullptr);
	const double st = std::strtod(tok[12].c_str(), nullptr);
	cpu_s = (hz > 0.0) ? (ut + st) / hz : -1.0;
	last_cpu = (int)std::strtol(tok[36].c_str(), nullptr, 10);

	return true;
}

#endif

static inline int tm_pinned_cpu(const char* status_path) {

#if defined(__linux__)

	FILE* f = std::fopen(status_path, "r");

	if (!f) {

		return -1;
	}

	char line[512];
	int only = -1;

	while (std::fgets(line, sizeof(line), f)) {

		if (std::strncmp(line, "Cpus_allowed_list:", 18) != 0) {

			continue;
		}

		char* p = line + 18;

		while (*p == ' ' || *p == '\t') {

			p++;
		}

		char* nl = std::strchr(p, '\n');

		if (nl) {

			*nl = '\0';
		}

		if (std::strchr(p, ',') || std::strchr(p, '-')) {

			only = -1;

		} else {

			only = (int)std::strtol(p, nullptr, 10);
		}

		break;
	}

	std::fclose(f);

	return only;

#else

	(void)status_path;

	return -1;

#endif
}

static inline std::vector<TidInfo> tm_scan_tids() {

	std::vector<TidInfo> out;

#if defined(__linux__)

	DIR* d = opendir("/proc/self/task");

	if (!d) {

		return out;
	}

	struct dirent* e = nullptr;

	while ((e = readdir(d)) != nullptr) {

		if (e->d_name[0] == '.') {

			continue;
		}

		const long tid = std::strtol(e->d_name, nullptr, 10);

		if (tid <= 0) {

			continue;
		}

		char pstat[256], pschd[256], pcomm[256], pstatus[256];
		std::snprintf(pstat, sizeof(pstat), "/proc/self/task/%ld/stat", tid);
		std::snprintf(pschd, sizeof(pschd), "/proc/self/task/%ld/schedstat", tid);
		std::snprintf(pcomm, sizeof(pcomm), "/proc/self/task/%ld/comm", tid);
		std::snprintf(pstatus, sizeof(pstatus), "/proc/self/task/%ld/status", tid);

		TidInfo ti;
		ti.tid = tid;
		ti.cpu_s = -1.0;
		ti.last_cpu = -1;
		ti.pinned_cpu = tm_pinned_cpu(pstatus);
		ti.runq_s = tm_read_runq_s(pschd);

		double cpu_s = -1.0;
		int last_cpu = -1;

		if (tm_parse_stat(pstat, cpu_s, last_cpu)) {

			ti.cpu_s = cpu_s;
			ti.last_cpu = last_cpu;
		}

		FILE* fc = std::fopen(pcomm, "r");

		if (fc) {

			char cb[128] = {0};

			if (std::fgets(cb, sizeof(cb), fc)) {

				char* nl = std::strchr(cb, '\n');

				if (nl) {

					*nl = '\0';
				}

				ti.comm = cb;
			}

			std::fclose(fc);
		}

		out.push_back(ti);
	}

	closedir(d);

#endif

	return out;
}

static inline PerfCounters tm_perf_open() {

	PerfCounters pc;
	pc.fd_cycles = -1;
	pc.fd_insns = -1;
	pc.fd_llc_ref = -1;
	pc.fd_llc_miss = -1;
	pc.fd_ref_cyc = -1;

#if defined(__linux__)

	pc.fd_cycles = tm_perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
	pc.fd_insns = tm_perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
	pc.fd_llc_ref = tm_perf_open_one(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
	pc.fd_llc_miss = tm_perf_open_one(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
	pc.fd_ref_cyc = tm_perf_open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES);

#endif

	return pc;
}

static inline long long tm_perf_read_one(int fd) {

#if defined(__linux__)

	if (fd < 0) {

		return -1;
	}

	unsigned long long v[3] = {0, 0, 0};

	if (read(fd, v, sizeof(v)) != (ssize_t)sizeof(v)) {

		return -1;
	}

	if (v[2] == 0) {

		return -1;
	}

	if (v[1] == v[2]) {

		return (long long)v[0];
	}

	return (long long)((double)v[0] * (double)v[1] / (double)v[2]);

#else

	(void)fd;

	return -1;

#endif
}

static inline void tm_perf_close(PerfCounters& pc) {

#if defined(__linux__)

	if (pc.fd_cycles >= 0) {

		close(pc.fd_cycles);
	}

	if (pc.fd_insns >= 0) {

		close(pc.fd_insns);
	}

	if (pc.fd_llc_ref >= 0) {

		close(pc.fd_llc_ref);
	}

	if (pc.fd_llc_miss >= 0) {

		close(pc.fd_llc_miss);
	}

	if (pc.fd_ref_cyc >= 0) {

		close(pc.fd_ref_cyc);
	}

#endif

	pc.fd_cycles = -1;
	pc.fd_insns = -1;
	pc.fd_llc_ref = -1;
	pc.fd_llc_miss = -1;
	pc.fd_ref_cyc = -1;
}

static inline ThreadSnap tm_snapshot(const PerfCounters& pc) {

	ThreadSnap s;
	s.compute_cpu_s = tm_thread_cpu_s();
	s.proc_cpu_s = tm_proc_cpu_s();
	s.runq_s = tm_thread_runq_s();
	tm_rusage_thread(s.nvcsw, s.nivcsw);
	s.cycles = tm_perf_read_one(pc.fd_cycles);
	s.insns = tm_perf_read_one(pc.fd_insns);
	s.llc_ref = tm_perf_read_one(pc.fd_llc_ref);
	s.llc_miss = tm_perf_read_one(pc.fd_llc_miss);
	s.ref_cycles = tm_perf_read_one(pc.fd_ref_cyc);

	return s;
}

static inline double tm_delta(double b, double a) {

	if (b < 0.0 || a < 0.0) {

		return -1.0;
	}

	return b - a;
}

static inline long long tm_delta_ll(long long b, long long a) {

	if (b < 0 || a < 0) {

		return -1;
	}

	return b - a;
}

static inline long tm_delta_l(long b, long a) {

	if (b < 0 || a < 0) {

		return -1;
	}

	return b - a;
}

#endif
