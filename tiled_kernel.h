#pragma once

// Templated tiled-GEMM kernel and launchers. Each registry kernel in
// trait_based_kernels.cu picks a Geometry + Load + Remap policy and
// instantiates `tiled_kernel`; the launchers below pick a TPB strategy.

#include "gemm_common.h"
#include "gemm_mainloop.h"

namespace tiled {

// ---- Load policies (gmem -> smem) ------------------------------------------
// Each policy provides:
//   static constexpr bool CLEAN          // selects clean vs OOB-tolerant store
//   template <int TILE_M,N,K, STRIDE_A,B, NT>
//   __device__ static void load(tid, rowBase, colBase, kOff,
//                                M, N, K, A, B, tileA, tileB);

// Scalar 4-byte loads with per-element OOB guards.
struct ScalarOOB {
    static constexpr bool CLEAN = false;

    template <int TILE_M, int TILE_N, int TILE_K, int STRIDE_A, int STRIDE_B, int NT>
    static __device__ __forceinline__ void load(
        int tid, int rowBase, int colBase, int kOff,
        int M, int N, int K,
        const float* __restrict__ A, const float* __restrict__ B,
        float (*tileA)[STRIDE_A], float (*tileB)[STRIDE_B]) {
        constexpr int LA = (TILE_M * TILE_K) / NT;
        constexpr int LB = (TILE_K * TILE_N) / NT;
        #pragma unroll
        for (int i = 0; i < LA; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / TILE_K, sC = idx % TILE_K;
            const int aR = rowBase + sR, aC = kOff + sC;
            tileA[sR][sC] = (aR < M && aC < K) ? A[aR * K + aC] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < LB; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / TILE_N, sC = idx % TILE_N;
            const int bR = kOff + sR, bC = colBase + sC;
            tileB[sR][sC] = (bR < K && bC < N) ? B[bR * N + bC] : 0.f;
        }
    }
};

// Scalar 4-byte loads, no bounds checks. Caller must enforce divisibility.
struct ScalarClean {
    static constexpr bool CLEAN = true;

    template <int TILE_M, int TILE_N, int TILE_K, int STRIDE_A, int STRIDE_B, int NT>
    static __device__ __forceinline__ void load(
        int tid, int rowBase, int colBase, int kOff,
        int /*M*/, int N, int K,
        const float* __restrict__ A, const float* __restrict__ B,
        float (*tileA)[STRIDE_A], float (*tileB)[STRIDE_B]) {
        constexpr int LA = (TILE_M * TILE_K) / NT;
        constexpr int LB = (TILE_K * TILE_N) / NT;
        #pragma unroll
        for (int i = 0; i < LA; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / TILE_K, sC = idx % TILE_K;
            tileA[sR][sC] = A[(rowBase + sR) * K + kOff + sC];
        }
        #pragma unroll
        for (int i = 0; i < LB; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / TILE_N, sC = idx % TILE_N;
            tileB[sR][sC] = B[(kOff + sR) * N + colBase + sC];
        }
    }
};

// LDG.128 (float4) loads. Caller must enforce divisibility *and* 16-byte
// alignment of (rowBase, kOff, colBase).
struct Vec4Clean {
    static constexpr bool CLEAN = true;

    template <int TILE_M, int TILE_N, int TILE_K, int STRIDE_A, int STRIDE_B, int NT>
    static __device__ __forceinline__ void load(
        int tid, int rowBase, int colBase, int kOff,
        int /*M*/, int N, int K,
        const float* __restrict__ A, const float* __restrict__ B,
        float (*tileA)[STRIDE_A], float (*tileB)[STRIDE_B]) {
        constexpr int LA4 = (TILE_M * TILE_K / 4) / NT;
        constexpr int LB4 = (TILE_K * TILE_N / 4) / NT;
        #pragma unroll
        for (int i = 0; i < LA4; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / (TILE_K / 4);
            const int sC = (idx % (TILE_K / 4)) * 4;
            const float4 v = *reinterpret_cast<const float4*>(&A[(rowBase + sR) * K + kOff + sC]);
            tileA[sR][sC + 0] = v.x; tileA[sR][sC + 1] = v.y;
            tileA[sR][sC + 2] = v.z; tileA[sR][sC + 3] = v.w;
        }
        #pragma unroll
        for (int i = 0; i < LB4; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / (TILE_N / 4);
            const int sC = (idx % (TILE_N / 4)) * 4;
            const float4 v = *reinterpret_cast<const float4*>(&B[(kOff + sR) * N + colBase + sC]);
            tileB[sR][sC + 0] = v.x; tileB[sR][sC + 1] = v.y;
            tileB[sR][sC + 2] = v.z; tileB[sR][sC + 3] = v.w;
        }
    }
};

// ---- Remap policies (blockIdx -> output tile (bx, by)) ---------------------

// 2D launch grid, blockIdx maps to output tile directly.
struct Linear {
    static constexpr bool IS_SUPERTILE = false;
    static __device__ __forceinline__ bool remap(
        int /*blocks_x*/, int /*blocks_y*/, int& bx, int& by) {
        bx = blockIdx.x;
        by = blockIdx.y;
        return true;
    }
};

// 1D launch grid; consecutive blocks visit a SUPER x SUPER cluster of
// output tiles before moving on (improves L2 reuse vs column-major).
template <int SUPER_>
struct Supertile {
    static constexpr bool IS_SUPERTILE = true;
    static constexpr int SUPER = SUPER_;
    static __device__ __forceinline__ bool remap(
        int blocks_x, int blocks_y, int& bx, int& by) {
        return gemm::supertile_remap<SUPER>(
            (int)blockIdx.x, blocks_x, blocks_y, bx, by);
    }
};

// ---- Kernel ----------------------------------------------------------------
// Traits expects:
//   TILE_M, TILE_N, TILE_K, STRIDE_A, STRIDE_B, BX, BY, ROWS, COLS, LB_BPS
//   using Load  = ScalarOOB | ScalarClean | Vec4Clean
//   using Remap = Linear | Supertile<SUPER>

template <class T, int TPB>
__global__ __launch_bounds__(T::BX * T::BY, T::LB_BPS)
void tiled_kernel(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float* __restrict__ C,
                  int M, int N, int K,
                  int blocks_x, int blocks_y) {
    constexpr int TILE_M = T::TILE_M, TILE_N = T::TILE_N, TILE_K = T::TILE_K;
    constexpr int STRIDE_A = T::STRIDE_A, STRIDE_B = T::STRIDE_B;
    constexpr int BX = T::BX, BY = T::BY, ROWS = T::ROWS, COLS = T::COLS;
    constexpr int NT = BX * BY;
    constexpr int TILE_A_ELEMS = TILE_M * STRIDE_A;
    using Load = typename T::Load;
    using Remap = typename T::Remap;

    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * BX + tx;

    int bx_remap, by_remap;
    if (!Remap::remap(blocks_x, blocks_y, bx_remap, by_remap)) return;

    extern __shared__ float smem[];
    float (*tileA)[STRIDE_A] = reinterpret_cast<float (*)[STRIDE_A]>(smem);
    float (*tileB)[STRIDE_B] = reinterpret_cast<float (*)[STRIDE_B]>(smem + TILE_A_ELEMS);

    const int numTiles = (K + TILE_K - 1) / TILE_K;

    for (int by_ = 0; by_ < TPB; ++by_)
    for (int bx_ = 0; bx_ < TPB; ++bx_) {
        const int rowBase = (by_remap * TPB + by_) * TILE_M;
        const int colBase = (bx_remap * TPB + bx_) * TILE_N;
        float sums[ROWS][COLS] = {{0.f}};

        for (int t = 0; t < numTiles; ++t) {
            Load::template load<TILE_M, TILE_N, TILE_K, STRIDE_A, STRIDE_B, NT>(
                tid, rowBase, colBase, t * TILE_K, M, N, K, A, B, tileA, tileB);
            __syncthreads();
            gemm::compute_inner<ROWS, COLS, TILE_K, STRIDE_A, STRIDE_B>(
                tileA, tileB, ty, tx, sums);
            __syncthreads();
        }
        if constexpr (Load::CLEAN)
            gemm::store_clean<ROWS, COLS>(C, sums, rowBase, colBase, ty, tx, N);
        else
            gemm::store_with_oob<ROWS, COLS>(C, sums, rowBase, colBase, ty, tx, M, N);
    }
}

// ---- Launch helpers --------------------------------------------------------

namespace detail {
template <class T>
constexpr int smem_bytes_v =
    (T::TILE_M * T::STRIDE_A + T::TILE_K * T::STRIDE_B) * sizeof(float);
}

// Linear remap, 2D grid. Picks TPB=2 if shape is divisible and grid >= 64
// blocks, else TPB=1.
template <class T>
inline void launch_2d(const GemmParams& p) {
    static_assert(!T::Remap::IS_SUPERTILE, "launch_2d requires Linear remap");
    constexpr int smem_bytes = detail::smem_bytes_v<T>;
    enable_dynamic_smem(tiled_kernel<T, 1>, smem_bytes);
    enable_dynamic_smem(tiled_kernel<T, 2>, smem_bytes);

    constexpr int e2m = T::TILE_M * 2, e2n = T::TILE_N * 2;
    const int g2x = (p.N + e2n - 1) / e2n;
    const int g2y = (p.M + e2m - 1) / e2m;
    const dim3 block(T::BX, T::BY);

    if ((p.M % e2m == 0) && (p.N % e2n == 0) && g2x * g2y >= 64) {
        tiled_kernel<T, 2><<<dim3(g2x, g2y), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g2x, g2y);
    } else {
        const int g1x = (p.N + T::TILE_N - 1) / T::TILE_N;
        const int g1y = (p.M + T::TILE_M - 1) / T::TILE_M;
        tiled_kernel<T, 1><<<dim3(g1x, g1y), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g1x, g1y);
    }
    CUDA_CHECK_LAST();
}

// Linear remap, 2D grid, three-way TPB (4 -> 2 -> 1). Used by tpb4.
template <class T>
inline void launch_2d_tpb4(const GemmParams& p) {
    static_assert(!T::Remap::IS_SUPERTILE, "launch_2d_tpb4 requires Linear remap");
    constexpr int smem_bytes = detail::smem_bytes_v<T>;
    enable_dynamic_smem(tiled_kernel<T, 1>, smem_bytes);
    enable_dynamic_smem(tiled_kernel<T, 2>, smem_bytes);
    enable_dynamic_smem(tiled_kernel<T, 4>, smem_bytes);

    constexpr int e4m = T::TILE_M * 4, e4n = T::TILE_N * 4;
    constexpr int e2m = T::TILE_M * 2, e2n = T::TILE_N * 2;
    const int g4x = (p.N + e4n - 1) / e4n, g4y = (p.M + e4m - 1) / e4m;
    const int g2x = (p.N + e2n - 1) / e2n, g2y = (p.M + e2m - 1) / e2m;
    const dim3 block(T::BX, T::BY);

    if ((p.M % e4m == 0) && (p.N % e4n == 0) && g4x * g4y >= 64) {
        tiled_kernel<T, 4><<<dim3(g4x, g4y), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g4x, g4y);
    } else if ((p.M % e2m == 0) && (p.N % e2n == 0) && g2x * g2y >= 64) {
        tiled_kernel<T, 2><<<dim3(g2x, g2y), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g2x, g2y);
    } else {
        const int g1x = (p.N + T::TILE_N - 1) / T::TILE_N;
        const int g1y = (p.M + T::TILE_M - 1) / T::TILE_M;
        tiled_kernel<T, 1><<<dim3(g1x, g1y), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g1x, g1y);
    }
    CUDA_CHECK_LAST();
}

// Supertile remap, 1D grid. TPB=2 if shape allows and grid is large enough.
template <class T>
inline void launch_supertile(const GemmParams& p) {
    static_assert(T::Remap::IS_SUPERTILE, "launch_supertile requires Supertile<> remap");
    constexpr int smem_bytes = detail::smem_bytes_v<T>;
    enable_dynamic_smem(tiled_kernel<T, 1>, smem_bytes);
    enable_dynamic_smem(tiled_kernel<T, 2>, smem_bytes);

    constexpr int e2m = T::TILE_M * 2, e2n = T::TILE_N * 2;
    const int g2x = (p.N + e2n - 1) / e2n;
    const int g2y = (p.M + e2m - 1) / e2m;
    const dim3 block(T::BX, T::BY);

    if ((p.M % e2m == 0) && (p.N % e2n == 0) && g2x * g2y >= 64) {
        tiled_kernel<T, 2><<<dim3(g2x * g2y, 1), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g2x, g2y);
    } else {
        const int g1x = (p.N + T::TILE_N - 1) / T::TILE_N;
        const int g1y = (p.M + T::TILE_M - 1) / T::TILE_M;
        tiled_kernel<T, 1><<<dim3(g1x * g1y, 1), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g1x, g1y);
    }
    CUDA_CHECK_LAST();
}

// Supertile remap, 1D grid, TPB=1, padded to sxc*syc*SUPER^2 blocks so the
// remap covers every (bx, by) position even at small grids.
template <class T>
inline void launch_supertile_padded(const GemmParams& p) {
    static_assert(T::Remap::IS_SUPERTILE, "launch_supertile_padded requires Supertile<> remap");
    constexpr int smem_bytes = detail::smem_bytes_v<T>;
    constexpr int SUPER = T::Remap::SUPER;
    enable_dynamic_smem(tiled_kernel<T, 1>, smem_bytes);

    const int g1x = (p.N + T::TILE_N - 1) / T::TILE_N;
    const int g1y = (p.M + T::TILE_M - 1) / T::TILE_M;
    const int sxc = (g1x + SUPER - 1) / SUPER;
    const int syc = (g1y + SUPER - 1) / SUPER;
    const int total = sxc * syc * SUPER * SUPER;

    tiled_kernel<T, 1><<<dim3(total, 1), dim3(T::BX, T::BY), smem_bytes>>>(
        p.dA, p.dB, p.dC, p.M, p.N, p.K, g1x, g1y);
    CUDA_CHECK_LAST();
}

} // namespace tiled
