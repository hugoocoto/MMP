#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static std::atomic<bool> g_stop{false};

static void on_signal(int) {

	g_stop.store(true, std::memory_order_relaxed);
}

static const char* arg_str(int argc, char** argv, const char* key, const char* def) {

	const size_t klen = std::strlen(key);

	for (int i = 1; i < argc; i++) {

		if (std::strncmp(argv[i], "--", 2) != 0) {

			continue;
		}

		const char* a = argv[i] + 2;

		if (std::strncmp(a, key, klen) == 0 && a[klen] == '=') {

			return a + klen + 1;
		}
	}

	return def;
}

static long arg_long(int argc, char** argv, const char* key, long def) {

	const char* v = arg_str(argc, argv, key, nullptr);

	if (!v || !*v) {

		return def;
	}

	return std::strtol(v, nullptr, 10);
}

static bool file_exists(const char* path) {

	struct stat st;

	return path && *path && stat(path, &st) == 0;
}

static void touch_file(const char* path) {

	if (!path || !*path) {

		return;
	}

	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);

	if (fd >= 0) {

		close(fd);
	}
}

static size_t probe_llc_bytes() {

	long v = 0;

#if defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)

	v = sysconf(_SC_LEVEL3_CACHE_SIZE);

#endif

#if defined(__APPLE__)

	size_t sz = 0;
	size_t len = sizeof(sz);

	if (sysctlbyname("hw.l3cachesize", &sz, &len, nullptr, 0) == 0 && sz > 0) {

		v = (long)sz;
	}

#endif

	if (v <= 0) {

		v = 32L << 20;
	}

	return (size_t)v;
}

struct Stopper {

	const char* stop_file;
	double max_seconds;
	std::chrono::steady_clock::time_point t0;

	bool should_stop() {

		if (g_stop.load(std::memory_order_relaxed)) {

			return true;
		}

		if (max_seconds > 0.0) {

			const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

			if (el >= max_seconds) {

				return true;
			}
		}

		return file_exists(stop_file);
	}
};

static int run_cpu(Stopper& stop, const char* ready_file) {

	volatile double sink = 0.0;
	double a = 1.0000001, b = 0.9999999, c = 1.1, d = 0.7, e = 0.3, f = 1.7;

	for (int k = 0; k < 8192; k++) {

		a = std::fma(a, b, c);
		d = std::fma(d, a, b);
		e = std::fma(e, d, a);
		f = std::fma(f, e, d);
		b = a * 0.9999999 + 1e-9;
		c = d * 1.0000001 - 1e-9;
	}

	sink += a + b + c + d + e + f;
	touch_file(ready_file);

	for (;;) {

		for (int batch = 0; batch < 256; batch++) {

			for (int k = 0; k < 8192; k++) {

				a = std::fma(a, b, c);
				d = std::fma(d, a, b);
				e = std::fma(e, d, a);
				f = std::fma(f, e, d);
				b = a * 0.9999999 + 1e-9;
				c = d * 1.0000001 - 1e-9;
			}

			sink += a + b + c + d + e + f;
		}

		if (stop.should_stop()) {

			break;
		}
	}

	return (int)(sink != 0.0);
}

static int run_cache(Stopper& stop, const char* ready_file, size_t bytes) {

	const size_t n = std::max<size_t>(bytes / sizeof(size_t), 1024);
	std::vector<size_t> next(n);
	std::vector<size_t> scratch(n, 0);
	std::vector<size_t> perm(n);

	std::iota(perm.begin(), perm.end(), (size_t)0);

	std::mt19937_64 rng(88172645463325252ULL);

	for (size_t i = n - 1; i > 0; i--) {

		size_t j = rng() % (i + 1);
		std::swap(perm[i], perm[j]);
	}

	for (size_t i = 0; i < n; i++) {

		next[perm[i]] = perm[(i + 1) % n];
	}

	size_t idx = 0;
	volatile size_t s = 0;

	for (size_t k = 0; k < n; k++) {

		idx = next[idx];
	}

	s += idx;
	touch_file(ready_file);

	bool stop_now = false;

	while (!stop_now) {

		for (size_t k = 0; k < n; k++) {

			idx = next[idx];
			scratch[idx] = idx;

			if ((k & 0xFFFFF) == 0 && stop.should_stop()) {

				stop_now = true;
				break;
			}
		}

		s += idx + scratch[idx];
	}

	return (int)(s != 0);
}

static int run_io(Stopper& stop, const char* ready_file, const char* scratch_path, size_t block) {

	std::string path = (scratch_path && *scratch_path) ? std::string(scratch_path) : std::string("/tmp/noise_daemon.io");
	int flags = O_RDWR | O_CREAT | O_TRUNC;

#if defined(__linux__) && defined(O_DIRECT)

	flags |= O_DIRECT;

#endif

	int fd = open(path.c_str(), flags, 0600);

	if (fd < 0 && (flags & (O_CREAT))) {

		flags = O_RDWR | O_CREAT | O_TRUNC;
		fd = open(path.c_str(), flags, 0600);
	}

	if (fd < 0) {

		std::fprintf(stderr, "noise_daemon io: cannot open %s\n", path.c_str());

		return 1;
	}

#if defined(__APPLE__)

	fcntl(fd, F_NOCACHE, 1);

#endif

	void* buf = nullptr;

	if (posix_memalign(&buf, 4096, block) != 0 || !buf) {

		std::fprintf(stderr, "noise_daemon io: posix_memalign failed\n");
		close(fd);

		return 1;
	}

	std::memset(buf, 0xA5, block);

	const off_t filesz = (off_t)(2ULL << 30);
	off_t off = 0;
	long cnt = 0;

	touch_file(ready_file);

	for (;;) {

		ssize_t w = pwrite(fd, buf, block, off);

		if (w < 0) {

			off = 0;
		}

		fsync(fd);

#if defined(__linux__)

		posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

#elif defined(__APPLE__)

		fcntl(fd, F_FULLFSYNC, 0);

#endif

		off += (off_t)block;

		if (off + (off_t)block >= filesz) {

			off = 0;
		}

		if (((++cnt) & 63) == 0) {

			char t[4096];
			ssize_t r = pread(fd, t, sizeof(t), 0);
			(void)r;
		}

		if (stop.should_stop()) {

			break;
		}
	}

	free(buf);
	close(fd);
	unlink(path.c_str());

	return 0;
}

int main(int argc, char** argv) {

	std::signal(SIGTERM, on_signal);
	std::signal(SIGINT, on_signal);

	const char* mode = arg_str(argc, argv, "mode", "cpu");
	const char* ready_file = arg_str(argc, argv, "ready-file", "");
	const char* stop_file = arg_str(argc, argv, "stop-file", "");
	const char* scratch_path = arg_str(argc, argv, "scratch", "");
	const double max_seconds = (double)arg_long(argc, argv, "max-seconds", 0);
	const long block = arg_long(argc, argv, "block", 1L << 20);

	size_t bytes = (size_t)arg_long(argc, argv, "bytes", 0);

	if (bytes == 0) {

		bytes = std::max<size_t>(probe_llc_bytes() * 3, (size_t)512 << 20);
	}

	std::fprintf(stderr, "noise_daemon pid=%d mode=%s bytes=%zu block=%ld max_seconds=%.0f\n", (int)getpid(), mode, bytes, block, max_seconds);

	Stopper stop{stop_file, max_seconds, std::chrono::steady_clock::now()};

	if (std::strcmp(mode, "cpu") == 0) {

		return run_cpu(stop, ready_file);
	}

	if (std::strcmp(mode, "cache") == 0) {

		return run_cache(stop, ready_file, bytes);
	}

	if (std::strcmp(mode, "io") == 0) {

		return run_io(stop, ready_file, scratch_path, (size_t)block);
	}

	std::fprintf(stderr, "noise_daemon: unknown mode '%s' (use cpu|cache|io)\n", mode);

	return 2;
}
