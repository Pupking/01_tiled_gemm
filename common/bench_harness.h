#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <string>
#include <utility>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(_err), __FILE__, __LINE__,           \
                         cudaGetErrorString(_err));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Kernel launches don't return cudaError_t; use this right after them.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Fill a device buffer with NaN (0xFF bytes = all-ones exponent, nonzero
// mantissa). Call before each verify run so a kernel that silently skips
// writes can't pass verify by leaving stale data behind.
template <typename T>
inline void poison_output(T* d_ptr, std::size_t n) {
    CUDA_CHECK(cudaMemset(d_ptr, 0xFF, n * sizeof(T)));
}

struct BenchStats {
    double median_ms;       // median per-iter latency across runs
    double mean_ms;
    double stddev_ms;       // sample stddev (N-1)
    double min_ms;
    double max_ms;
    int    runs;
    int    iters_per_run;
};

// Median + stddev across `runs` repeats; each repeat times `iters` launches
// with one cudaEventElapsedTime so we report a per-iter latency.
template <typename Fn>
inline BenchStats benchmark_kernel(Fn&& fn,
                                   int warmup_iters = 3,
                                   int iters        = 50,
                                   int runs         = 5) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup_iters; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples(runs);
    for (int r = 0; r < runs; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples[r] = ms / iters;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());

    BenchStats s{};
    s.runs          = runs;
    s.iters_per_run = iters;
    s.min_ms        = samples.front();
    s.max_ms        = samples.back();
    s.median_ms     = (runs % 2)
                      ? samples[runs / 2]
                      : 0.5 * (samples[runs / 2 - 1] + samples[runs / 2]);

    double mean = 0;
    for (double v : samples) mean += v;
    mean /= runs;
    s.mean_ms = mean;

    double var = 0;
    for (double v : samples) { double d = v - mean; var += d * d; }
    s.stddev_ms = (runs > 1) ? std::sqrt(var / (runs - 1)) : 0.0;

    return s;
}

inline float to_float(float x)  { return x; }
inline float to_float(__half x) { return __half2float(x); }

// Element-wise `|got - ref| <= atol + rtol * |ref|`. NaN in `got` always
// fails (the negated `<=` is true for NaN). Operates on host buffers.
template <typename T>
inline bool verify_close(const T* ref, const T* got, std::size_t n,
                         double atol, double rtol, int max_print = 5) {
    int printed = 0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double r   = static_cast<double>(to_float(ref[i]));
        double g   = static_cast<double>(to_float(got[i]));
        double tol = atol + rtol * std::fabs(r);
        double err = std::fabs(r - g);
        if (!(err <= tol)) {
            if (printed < max_print) {
                std::fprintf(stderr,
                    "  [%zu] ref=%g got=%g |err|=%g tol=%g\n",
                    i, r, g, err, tol);
                ++printed;
            }
            ++bad;
        }
    }
    if (bad) {
        std::fprintf(stderr,
            "verify FAILED: %zu / %zu elements diverged\n", bad, n);
        return false;
    }
    return true;
}

template <typename LaunchFn>
using KernelRegistry = std::vector<std::pair<std::string, LaunchFn>>;
