#pragma once

// Layer 0 — Tiled GEMM: common types, host helpers, and references.

#include "common/bench_harness.h"

#include <cublas_v2.h>
#include <cstdint>
#include <random>

// ---------------------------------------------------------------------------
// cuBLAS error check
// ---------------------------------------------------------------------------

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                 \
                         static_cast<int>(_s), __FILE__, __LINE__);            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Problem + launcher contract
// ---------------------------------------------------------------------------
// Row-major: C[M,N] = A[M,K] * B[K,N]. Every kernel launches via
// a function matching GemmLaunch. Registered in main.cu's KernelRegistry

struct GemmParams {
    int M, N, K;
    const float* dA;   // device, row-major [M, K]
    const float* dB;   // device, row-major [K, N]
    float*       dC;   // device, row-major [M, N]
};

using GemmLaunch = void (*)(const GemmParams&);

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

inline void fill_uniform(float* buf, std::size_t n,
                         float lo, float hi, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (std::size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

// ---------------------------------------------------------------------------
// cuBLAS reference
// ---------------------------------------------------------------------------
// cuBLAS is column-major. To compute row-major C = A * B we ask cuBLAS
// for the equivalent column-major C^T = B^T * A^T (swapping operands,
// both OP_N). Computed in FP32 regardless of hardware tensor-core path.

inline void cublas_gemm_fp32(cublasHandle_t handle, const GemmParams& p) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        p.N, p.M, p.K,
        &alpha,
        p.dB, CUDA_R_32F, p.N,
        p.dA, CUDA_R_32F, p.K,
        &beta,
        p.dC, CUDA_R_32F, p.N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}

// ---------------------------------------------------------------------------
// FP64 Kahan-summed CPU reference
// ---------------------------------------------------------------------------
// Slow (O(M*N*K) scalar loop). Used ONCE, for a small shape (M=N=K=128),
// to cross-validate that cuBLAS agrees with exact FP32-accumulated compute.
// All other verifications are against cuBLAS
// Refer(https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
inline void cpu_gemm_kahan_fp64(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0, comp = 0.0;                  // Kahan compensation
            for (int k = 0; k < K; ++k) {
                double val = double(A[i * K + k]) * double(B[k * N + j]);
                double y = val - comp;
                double t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

