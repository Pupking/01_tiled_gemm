#pragma once

// Problem definition, cuBLAS reference, and host helpers shared by every
// kernel TU.

#include "common/bench_harness.h"

#include <cublas_v2.h>
#include <cstdint>
#include <random>

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                 \
                         static_cast<int>(_s), __FILE__, __LINE__);            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Row-major C[M,N] = A[M,K] * B[K,N]. Every kernel ships a launcher with
// this signature; main.cu's KernelRegistry holds the (name, launcher) pairs.
struct GemmParams {
    int M, N, K;
    const float* dA;   // device, row-major [M, K]
    const float* dB;   // device, row-major [K, N]
    float*       dC;   // device, row-major [M, N]
};
using GemmLaunch = void (*)(const GemmParams&);

inline void fill_uniform(float* buf, std::size_t n,
                         float lo, float hi, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (std::size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

// cuBLAS is column-major; for row-major C = A * B we ask for the equivalent
// column-major C^T = B^T * A^T (operands swapped, both OP_N).
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

// Opt a kernel into >48 KB dynamic SMEM. Idempotent; cheap enough to call
// once per launch rather than caching (the previous static-bool guard was
// shared across template instantiations and silently skipped the opt-in).
template <typename Kernel>
inline void enable_dynamic_smem(Kernel kernel, int smem_bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));
}

// Reference-of-reference: O(M*N*K) FP64 with Kahan compensation. Used once
// at the smallest shape to cross-validate cuBLAS itself.
//   https://en.wikipedia.org/wiki/Kahan_summation_algorithm
inline void cpu_gemm_kahan_fp64(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0, comp = 0.0;
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
