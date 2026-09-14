#pragma once

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#ifndef BENCH_CSV

	#define BENCH_CSV 0

#endif

[[maybe_unused]] inline long parse_arg_long(int argc, char** argv, const char* key, long default_val) {

	for (int i = 1; i < argc; i++) {

		const char* arg = argv[i];

		if (arg[0] != '-' || arg[1] != '-') {

			continue;
		}

		const char* p = arg + 2;
		const char* q = key;

		while (*q && *p == *q) {
			p++;
			q++;
		}

		if (*q == '\0' && *p == '=') {

			char* end = nullptr;
			long v = std::strtol(p + 1, &end, 10);

			if (end != p + 1) {

				return v;
			}
		}
	}

	return default_val;
}

[[maybe_unused]] inline double parse_arg_double(int argc, char** argv, const char* key, double default_val) {

	for (int i = 1; i < argc; i++) {

		const char* arg = argv[i];

		if (arg[0] != '-' || arg[1] != '-') {

			continue;
		}

		const char* p = arg + 2;
		const char* q = key;

		while (*q && *p == *q) {
			p++;
			q++;
		}

		if (*q == '\0' && *p == '=') {

			char* end = nullptr;
			double v = std::strtod(p + 1, &end);

			if (end != p + 1) {

				return v;
			}
		}
	}

	return default_val;
}

[[maybe_unused]] inline const char* parse_arg_str(int argc, char** argv, const char* key, const char* default_val) {

	for (int i = 1; i < argc; i++) {

		const char* arg = argv[i];

		if (arg[0] != '-' || arg[1] != '-') {

			continue;
		}

		const char* p = arg + 2;
		const char* q = key;

		while (*q && *p == *q) {
			p++;
			q++;
		}

		if (*q == '\0' && *p == '=') {

			return p + 1;
		}
	}

	return default_val;
}

[[maybe_unused]] inline useconds_t parse_env_uint(const char* var, useconds_t default_val) {

	const char* env = std::getenv(var);

	if (!env || !*env) {

		return default_val;
	}

	char* end = nullptr;
	long v = std::strtol(env, &end, 10);

	return (end == env || v <= 0) ? default_val : (useconds_t)v;
}

[[maybe_unused]] inline useconds_t example_delay_us(useconds_t default_us) {

	#if BENCH_CSV

	(void)default_us;
	return 0;

	#else

	return parse_env_uint("MAL_EXAMPLE_DELAY_US", default_us);

	#endif
}

[[maybe_unused]] inline useconds_t sparse_delay_scale_percent() {

	#if BENCH_CSV

	return 0;

	#else

	return parse_env_uint("MAL_SPARSE_DELAY_SCALE_PERCENT", 100);

	#endif
}

[[maybe_unused]] inline void print_bench_csv(const char* example, const char* variant, const char* mode, int np, int np_active, long work_items, double seconds, int errors) {

	#if BENCH_CSV

	std::printf("CSV,%s,%s,%s,%d,%d,%ld,%.6f,%d\n", example, variant, mode, np, np_active, work_items, seconds, errors);

	#else

	(void)example;
	(void)variant;
	(void)mode;
	(void)np;
	(void)np_active;
	(void)work_items;
	(void)seconds;
	(void)errors;

	#endif
}
