// Layer 0 driver. Registers one kernel per variant, verifies each against
// cuBLAS (reference), benchmarks with median+stddev, prints a results table.
//
// --cross-check runs a one-time cuBLAS vs Kahan-FP64 sanity check at 128^3

#include "common/bench_harness.h"
#include "gemm_common.h"
#include "launchers.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
    int M = 512;
    int N = 512;
    int K = 512;
    int warmup = 3;
    int iters = 20;
    int runs = 5;
    unsigned seed = 0xC0FFEEu;
    std::string kernel = "all";
    bool cross_check = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if      (k == "--M")            a.M      = std::atoi(next("--M"));
        else if (k == "--N")            a.N      = std::atoi(next("--N"));
        else if (k == "--K")            a.K      = std::atoi(next("--K"));
        else if (k == "--warmup")       a.warmup = std::atoi(next("--warmup"));
        else if (k == "--iters")        a.iters  = std::atoi(next("--iters"));
        else if (k == "--runs")         a.runs   = std::atoi(next("--runs"));
        else if (k == "--seed")         a.seed   = (unsigned)std::strtoul(next("--seed"), nullptr, 0);
        else if (k == "--kernel")       a.kernel = next("--kernel");
        else if (k == "--cross-check") a.cross_check = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); std::exit(2); }
    }
    return a;
}

double gflops_of(int M, int N, int K, double median_ms) {
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    return (flops / 1.0e9) / (median_ms / 1.0e3);
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // --- Optional one-time reference-of-reference check ----------------------
    if (args.cross_check) {
        const int S = 128;
        std::vector<float> hA(S * S), hB(S * S), hC_cublas(S * S), hC_kahan(S * S);
        fill_uniform(hA.data(), hA.size(), -1.0f, 1.0f, args.seed ^ 0x1u);
        fill_uniform(hB.data(), hB.size(), -1.0f, 1.0f, args.seed ^ 0x2u);


        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, S * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, S * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, S * S * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), S * S * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), S * S * sizeof(float), cudaMemcpyHostToDevice));

        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        GemmParams cp{S, S, S, dA, dB, dC};
        cublas_gemm_fp32(h, cp);
        CUDA_CHECK(cudaMemcpy(hC_cublas.data(), dC, S * S * sizeof(float), cudaMemcpyDeviceToHost));
        CUBLAS_CHECK(cublasDestroy(h));

        cpu_gemm_kahan_fp64(hA.data(), hB.data(), hC_kahan.data(), S, S, S);

        const bool ok = verify_close<float>(hC_kahan.data(), hC_cublas.data(),
                                            S * S, 1e-4f, 1e-5f);
        std::printf("cross-check cuBLAS vs Kahan FP64 (M=N=K=%d): %s\n", S, ok ? "OK" : "FAIL");
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        if (!ok) return 1;
    }

    // --- Problem setup --------------------------------------------------------
    const int M = args.M, N = args.N, K = args.K;
    const size_t sA = (size_t)M * K, sB = (size_t)K * N, sC = (size_t)M * N;

    std::vector<float> hA(sA), hB(sB), hRef(sC), hGot(sC);
    fill_uniform(hA.data(), hA.size(), -1.0f, 1.0f, args.seed ^ 0xA1u);
    fill_uniform(hB.data(), hB.size(), -1.0f, 1.0f, args.seed ^ 0xB2u);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, sB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, sC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sB * sizeof(float), cudaMemcpyHostToDevice));

    // --- Reference on this exact shape via cuBLAS -----------------------------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    GemmParams p{M, N, K, dA, dB, dC};
    cublas_gemm_fp32(handle, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, sC * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Kernel registry ------------------------------------------------------
    KernelRegistry<GemmLaunch> registry;
    registry.emplace_back("naive",           naive_launch);
    registry.emplace_back("tiled",           tiled_launch);
    registry.emplace_back("tiled_coalesced", tiled_coalesced_launch);
    registry.emplace_back("regblock",        regblock_launch);
    registry.emplace_back("warp_rebalance",  warp_rebalance_launch);
    registry.emplace_back("bank_pad_vec", bank_pad_vec_launch);
    registry.emplace_back("gmem_vec",     gmem_vec_launch);
    registry.emplace_back("tpb4",         tpb4_launch);
    registry.emplace_back("k32_tpb4",     k32_tpb4_launch);
    registry.emplace_back("sweep_A_16x8_8x4",   sweep_A_launch);
    registry.emplace_back("sweep_B_16x16_4x4",  sweep_B_launch);
    registry.emplace_back("sweep_C_8x16_4x8",   sweep_C_launch);
    registry.emplace_back("sweep_D_8x8_8x8",    sweep_D_launch);
    registry.emplace_back("sweep_E_32x16_4x2",  sweep_E_launch);
    registry.emplace_back("sweep_F_32x8_8x2",   sweep_F_launch);
    registry.emplace_back("splitk1",      splitk1_launch);
    registry.emplace_back("splitk2",      splitk2_launch);
    registry.emplace_back("splitk3",      splitk3_launch);
    registry.emplace_back("splitk4",      splitk4_launch);
    registry.emplace_back("multibuf2",    multibuf2_launch);
    registry.emplace_back("multibuf_b2",  multibuf_b2_launch);
    registry.emplace_back("multibuf_b3",  multibuf_b3_launch);
    registry.emplace_back("multibuf_b4",  multibuf_b4_launch);
    registry.emplace_back("wider_m",      wider_m_launch);
    registry.emplace_back("tile128",      tile128_launch);
    registry.emplace_back("wider_n",      wider_n_launch);
    registry.emplace_back("supertile",    supertile_launch);
    registry.emplace_back("drop_oob",     drop_oob_launch);
    registry.emplace_back("super_wider",  super_wider_launch);
    registry.emplace_back("ldg_vec",      ldg_vec_launch);
    registry.emplace_back("ldg_super",    ldg_super_launch);
    registry.emplace_back("ldg_async",    ldg_async_launch);
    registry.emplace_back("ldg_super_big",ldg_super_big_launch);
    registry.emplace_back("ldg_k128",     ldg_k128_launch);
    registry.emplace_back("ldg_wider2",   ldg_wider2_launch);
    registry.emplace_back("ldg_max",      ldg_max_launch);

    // Compute a per-shape absolute tolerance floor so rtol dominates for
    // typical values but we still catch drift near zero.
    const float atol = 1e-4f;
    const float rtol = 1e-3f;

    // Time cuBLAS up front so every per-kernel row can print its % of cuBLAS.
    // Keep stats around to print as the trailing "cublas" reference row.
    BenchStats cublas_stats = benchmark_kernel(
        [&]() {
            poison_output(dC, sC);
            cublas_gemm_fp32(handle, p);
        },
        args.warmup, args.iters, args.runs);

    std::printf("%-16s  %10s  %10s  %7s  %8s  %9s  %9s  %s\n",
                "kernel", "median(ms)", "stddev(ms)", "min(ms)", "GFLOPS", "vs naive", "% cublas", "verify");
    std::printf("%-16s  %10s  %10s  %7s  %8s  %9s  %9s  %s\n",
                "----------------", "----------", "----------", "-------", "------", "---------", "---------", "------");

    // Captured from the naive row so every later row can print its speedup.
    // -1 means "not yet seen" (e.g. user ran --kernel <not naive>).
    double naive_median_ms = -1.0;
    auto fmt_speedup = [&](double median_ms) -> std::string {
        char buf[16];
        if (naive_median_ms <= 0.0) {
            std::snprintf(buf, sizeof(buf), "%9s", "--");
        } else {
            std::snprintf(buf, sizeof(buf), "%8.2fx", naive_median_ms / median_ms);
        }
        return std::string(buf);
    };
    // % of cuBLAS performance = (GFLOPS / cuBLAS_GFLOPS) * 100 = cuBLAS_ms / kernel_ms * 100.
    // Higher is better; cuBLAS itself is 100 %.
    auto fmt_pct_cublas = [&](double median_ms) -> std::string {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%8.1f%%", 100.0 * cublas_stats.median_ms / median_ms);
        return std::string(buf);
    };

    for (auto& entry : registry) {
        const std::string& name = entry.first;
        GemmLaunch launch = entry.second;
        if (args.kernel != "all" && args.kernel != name) continue;


        // 1. Verify on a fresh (poisoned) output.
        poison_output(dC, sC);
        launch(p);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hGot.data(), dC, sC * sizeof(float), cudaMemcpyDeviceToHost));
        const bool pass = verify_close<float>(hRef.data(), hGot.data(), sC, atol, rtol);

        // 2. Benchmark (poison before each run inside the lambda so we also
        //    catch kernels that silently skip writes).
        BenchStats stats = benchmark_kernel(
            [&]() {
                poison_output(dC, sC);
                launch(p);
            },
            args.warmup, args.iters, args.runs);

        if (name == "naive") naive_median_ms = stats.median_ms;

        std::printf("%-16s  %10.4f  %10.4f  %7.4f  %8.2f  %s  %s  %s\n",
                    name.c_str(),
                    stats.median_ms, stats.stddev_ms, stats.min_ms,
                    gflops_of(M, N, K, stats.median_ms),
                    fmt_speedup(stats.median_ms).c_str(),
                    fmt_pct_cublas(stats.median_ms).c_str(),
                    pass ? "PASS" : "FAIL");
    }

    if (args.kernel == "all" || args.kernel == "cublas") {
        std::printf("%-16s  %10.4f  %10.4f  %7.4f  %8.2f  %s  %s  %s\n",
                    "cublas",
                    cublas_stats.median_ms, cublas_stats.stddev_ms, cublas_stats.min_ms,
                    gflops_of(M, N, K, cublas_stats.median_ms),
                    fmt_speedup(cublas_stats.median_ms).c_str(),
                    fmt_pct_cublas(cublas_stats.median_ms).c_str(),
                    "ref");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}

