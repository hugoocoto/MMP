#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct CsrMatrix {

	long rows{0};
	long cols{0};
	long nnz{0};
	std::vector<long> row_ptr;
	std::vector<int> col_idx;
	std::vector<float> vals;

	long max_row_nnz() const {

		long m = 0;

		for (long r = 0; r < rows; r++) {

			m = std::max(m, row_ptr[(size_t)r + 1] - row_ptr[(size_t)r]);
		}

		return m;
	}

	double mean_row_nnz() const {

		return rows > 0 ? (double)nnz / (double)rows : 0.0;
	}
};

namespace mtxdetail {

struct Coo {
	int r;
	int c;
	float v;
};

inline void build_csr_from_coo(long rows, long cols, std::vector<Coo>& coo, CsrMatrix& out) {

	out.rows = rows;
	out.cols = cols;
	out.nnz = (long)coo.size();
	out.row_ptr.assign((size_t)rows + 1, 0);

	for (const auto& e : coo) {

		out.row_ptr[(size_t)e.r + 1]++;
	}

	for (long r = 0; r < rows; r++) {

		out.row_ptr[(size_t)r + 1] += out.row_ptr[(size_t)r];
	}

	out.col_idx.assign((size_t)out.nnz, 0);
	out.vals.assign((size_t)out.nnz, 0.0f);

	std::vector<long> cursor(out.row_ptr.begin(), out.row_ptr.end() - 1);

	for (const auto& e : coo) {

		const long pos = cursor[(size_t)e.r]++;
		out.col_idx[(size_t)pos] = e.c;
		out.vals[(size_t)pos] = e.v;
	}
}

}

inline bool load_matrix_market(const std::string& path, CsrMatrix& out, std::string& err) {

	std::ifstream in(path);

	if (!in) {
		err = "cannot open file: " + path;
		return false;
	}

	std::string line;

	if (!std::getline(in, line)) {
		err = "empty file";
		return false;
	}

	bool is_pattern = false, is_symmetric = false, is_skew = false;
	{

		std::string tag, object, format, field, symmetry;
		std::istringstream bs(line);
		bs >> tag >> object >> format >> field >> symmetry;
		auto low = [](std::string s) { for (auto& c : s){ c = (char)std::tolower((unsigned char)c);
} return s; };
		tag = low(tag);
		object = low(object);
		format = low(format);
		field = low(field);
		symmetry = low(symmetry);

		if (tag.rfind("%%matrixmarket", 0) != 0) {
			err = "missing %%MatrixMarket banner";
			return false;
		}

		if (format != "coordinate") {
			err = "only 'coordinate' (sparse) format supported, got: " + format;
			return false;
		}

		if (field == "complex" || symmetry == "hermitian") {
			err = "complex/hermitian matrices not supported";
			return false;
		}
		is_pattern = (field == "pattern");
		is_symmetric = (symmetry == "symmetric");
		is_skew = (symmetry == "skew-symmetric");
	}

	long M = 0, N = 0, declared_nnz = 0;
	{

		bool got = false;

		while (std::getline(in, line)) {

			if (line.empty() || line[0] == '%') {

				continue;
			}

			std::istringstream hs(line);

			if (!(hs >> M >> N >> declared_nnz)) {
				err = "malformed size header";
				return false;
			}
			got = true;
			break;
		}

		if (!got) {
			err = "missing size header";
			return false;
		}

		if (M <= 0 || N <= 0 || declared_nnz < 0) {
			err = "non-positive dimensions";
			return false;
		}

		if (M > INT_MAX || N > INT_MAX) {
			err = "matrix dimension exceeds int index range";
			return false;
		}
	}

	std::vector<mtxdetail::Coo> coo;
	coo.reserve((size_t)declared_nnz * (is_symmetric || is_skew ? 2 : 1));

	long read = 0;

	while (read < declared_nnz && std::getline(in, line)) {

		if (line.empty() || line[0] == '%') {

			continue;
		}

		std::istringstream rs(line);
		long r = 0, c = 0;
		double v = 1.0;

		if (!(rs >> r >> c)) {
			err = "malformed entry at line for nnz " + std::to_string(read);
			return false;
		}

		if (!is_pattern) { rs >> v; }
		read++;

		r--;
		c--;

		if (r < 0 || r >= M || c < 0 || c >= N) {
			err = "index out of range";
			return false;
		}

		coo.push_back({(int)r, (int)c, (float)v});

		if ((is_symmetric || is_skew) && r != c) {

			coo.push_back({(int)c, (int)r, (float)(is_skew ? -v : v)});
		}
	}

	if (read != declared_nnz) {
		err = "truncated file: expected " + std::to_string(declared_nnz) + " entries, got " + std::to_string(read);
		return false;
	}

	mtxdetail::build_csr_from_coo(M, N, coo, out);
	return true;
}

inline CsrMatrix generate_powerlaw(long n, long avg_deg, unsigned int seed, double s = 1.0) {

	CsrMatrix out;
	out.rows = n;
	out.cols = n;
	out.row_ptr.assign((size_t)n + 1, 0);

	double harmonic = 0.0;

	for (long r = 0; r < n; r++) {

		harmonic += 1.0 / std::pow((double)(r + 1), s);
	}

	const double scale = (double)avg_deg * (double)n / harmonic;

	std::vector<int> degree((size_t)n);

	for (long r = 0; r < n; r++) {

		long d = (long)std::llround(scale / std::pow((double)(r + 1), s));

		if (d < 1) {

			d = 1;
		}

		if (d > n) {

			d = n;
		}

		degree[(size_t)r] = (int)d;
		out.row_ptr[(size_t)r + 1] = out.row_ptr[(size_t)r] + d;
	}

	out.nnz = out.row_ptr[(size_t)n];
	out.col_idx.assign((size_t)out.nnz, 0);
	out.vals.assign((size_t)out.nnz, 1.0f);

	unsigned int st = seed ? seed : 1u;
	auto next = [&st]() { st = st * 1664525u + 1013904223u; return st; };

	for (long r = 0; r < n; r++) {

		const long base = out.row_ptr[(size_t)r];
		const int d = degree[(size_t)r];

		for (int k = 0; k < d; k++) {

			out.col_idx[(size_t)(base + k)] = (int)(next() % (unsigned long)n);
			out.vals[(size_t)(base + k)] = 0.5f + (float)(next() % 100u) * 0.01f;
		}
	}

	return out;
}

inline CsrMatrix generate_clustered(long n, long avg_deg, unsigned int seed, double cluster_frac = 0.25, double contrast = 7.0) {

	CsrMatrix out;
	out.rows = n;
	out.cols = n;
	out.row_ptr.assign((size_t)n + 1, 0);

	if (cluster_frac < 0.0) {

		cluster_frac = 0.0;
	}

	if (cluster_frac > 1.0) {

		cluster_frac = 1.0;
	}

	if (contrast < 1.0) {

		contrast = 1.0;
	}

	const double denom = cluster_frac * contrast + (1.0 - cluster_frac);
	double d_l = (double)avg_deg / (denom > 0.0 ? denom : 1.0);
	double d_h = contrast * d_l;
	const long heavy_rows = (long)std::llround(cluster_frac * (double)n);

	std::vector<int> degree((size_t)n);

	for (long r = 0; r < n; r++) {

		long d = (long)std::llround(r < heavy_rows ? d_h : d_l);

		if (d < 1) {

			d = 1;
		}

		if (d > n) {

			d = n;
		}

		degree[(size_t)r] = (int)d;
		out.row_ptr[(size_t)r + 1] = out.row_ptr[(size_t)r] + d;
	}

	out.nnz = out.row_ptr[(size_t)n];
	out.col_idx.assign((size_t)out.nnz, 0);
	out.vals.assign((size_t)out.nnz, 1.0f);

	unsigned int st = seed ? seed : 1u;
	auto next = [&st]() { st = st * 1664525u + 1013904223u; return st; };

	for (long r = 0; r < n; r++) {

		const long base = out.row_ptr[(size_t)r];
		const int d = degree[(size_t)r];

		for (int k = 0; k < d; k++) {

			out.col_idx[(size_t)(base + k)] = (int)(next() % (unsigned long)n);
			out.vals[(size_t)(base + k)] = 0.5f + (float)(next() % 100u) * 0.01f;
		}
	}

	return out;
}

inline CsrMatrix generate_spike(long n, long avg_deg, unsigned int seed, double contrast, long n_spikes) {

	CsrMatrix out;
	out.rows = n;
	out.cols = n;
	out.row_ptr.assign((size_t)n + 1, 0);

	if (contrast < 1.0) {

		contrast = 1.0;
	}

	if (n_spikes < 1) {

		n_spikes = 1;
	}

	long spike_deg = (long)std::llround((double)avg_deg * contrast);

	if (spike_deg < 1) {

		spike_deg = 1;
	}

	if (spike_deg > n) {

		spike_deg = n;
	}

	const long total_target = n * avg_deg;
	long light_deg = 1;

	if (n - n_spikes > 0) {

		double dl = (double)(total_target - n_spikes * spike_deg) / (double)(n - n_spikes);
		light_deg = (long)std::llround(dl);

		if (light_deg < 1) {

			light_deg = 1;
		}
	}

	std::vector<char> is_spike((size_t)n, 0);

	for (long s = 0; s < n_spikes; s++) {

		long pos = (long)((double)(s + 1) * (double)n / (double)(n_spikes + 1));

		if (pos < 0) {

			pos = 0;
		}

		if (pos >= n) {

			pos = n - 1;
		}

		is_spike[(size_t)pos] = 1;
	}

	std::vector<int> degree((size_t)n);

	for (long r = 0; r < n; r++) {

		long d = is_spike[(size_t)r] ? spike_deg : light_deg;

		if (d < 1) {

			d = 1;
		}

		if (d > n) {

			d = n;
		}

		degree[(size_t)r] = (int)d;
		out.row_ptr[(size_t)r + 1] = out.row_ptr[(size_t)r] + d;
	}

	out.nnz = out.row_ptr[(size_t)n];
	out.col_idx.assign((size_t)out.nnz, 0);
	out.vals.assign((size_t)out.nnz, 1.0f);

	unsigned int st = seed ? seed : 1u;
	auto next = [&st]() { st = st * 1664525u + 1013904223u; return st; };

	for (long r = 0; r < n; r++) {

		const long base = out.row_ptr[(size_t)r];
		const int d = degree[(size_t)r];

		for (int k = 0; k < d; k++) {

			out.col_idx[(size_t)(base + k)] = (int)(next() % (unsigned long)n);
			out.vals[(size_t)(base + k)] = 0.5f + (float)(next() % 100u) * 0.01f;
		}
	}

	return out;
}

inline CsrMatrix build_synthetic(const char* pattern, long n, long avg_deg, unsigned int seed, double contrast, double cluster_frac, long n_spikes = 4) {

	if (pattern && std::strcmp(pattern, "uniform") == 0) {

		return generate_clustered(n, avg_deg, seed, cluster_frac, 1.0);
	}

	if (pattern && std::strcmp(pattern, "spike") == 0) {

		return generate_spike(n, avg_deg, seed, contrast, n_spikes);
	}

	return generate_clustered(n, avg_deg, seed, cluster_frac, contrast);
}
