#pragma once

// Per-thread compute and store helpers shared by the COLS=4 register-block
// family of kernels (wider_m / supertile / drop_oob / ldg_vec / ldg_super /
// ldg_super_big / ldg_k128).
//
// Why these and not the loads:
//   - The compute reduction (LDS A -> LDS.128 B -> FMA outer product) is
//     identical across all of these and contributes no per-file insight.
//   - The store (STG.128 with optional OOB fallback) is also identical.
//   - The *loads* differ per file (scalar vs LDG.128 vs cp.async, OOB or
//     not, supertile remap) - those are what each step is testing, so they
//     stay inline in each kernel file where they're easy to compare.
//
// All helpers take SMEM tile pointers shaped as `T (*)[STRIDE]` to work with
// both `__shared__ T tile[M][STRIDE]` (decays) and dynamic-SMEM
// `reinterpret_cast<T (*)[STRIDE]>(smem)` casts.

#include <cuda_runtime.h>

namespace gemm {

// FMA reduction for one TILE_K-deep slice.
// Each thread reads ROWS scalars from tileA's column k (broadcast across the
// thread's COLS) and one float4 from tileB's row k, then accumulates ROWSxCOLS
// FMAs into `sums`. STRIDE_A and STRIDE_B are the column counts of the
// SMEM tiles (may include +1 padding to break LDS bank conflicts).
template <int ROWS, int COLS, int TILE_K, int STRIDE_A, int STRIDE_B>
__device__ __forceinline__ void compute_inner(
    const float (*tileA)[STRIDE_A],
    const float (*tileB)[STRIDE_B],
    int ty, int tx,
    float (&sums)[ROWS][COLS]) {
    static_assert(COLS == 4, "compute_inner specialised for COLS=4 (LDS.128 b-load)");

    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
        float a[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
            a[r] = tileA[ty * ROWS + r][k];

        const float4 bv = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS]);
        const float b[COLS] = { bv.x, bv.y, bv.z, bv.w };

        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            #pragma unroll
            for (int c = 0; c < COLS; ++c)
                sums[r][c] += a[r] * b[c];
        }
    }
}

// STG.128 store with no bounds checks. Caller must guarantee row+ROWS-1 < M
// and col+COLS-1 < N (the launcher's clean-shape gate enforces this).
template <int ROWS, int COLS>
__device__ __forceinline__ void store_clean(
    float* __restrict__ C,
    const float (&sums)[ROWS][COLS],
    int rowBase, int colBase, int ty, int tx,
    int N) {
    static_assert(COLS == 4, "store_clean specialised for COLS=4 (STG.128)");

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = rowBase + ty * ROWS + r;
        const int col = colBase + tx * COLS;
        const float4 out = { sums[r][0], sums[r][1], sums[r][2], sums[r][3] };
        *reinterpret_cast<float4*>(&C[row * N + col]) = out;
    }
}

// STG.128 with per-row OOB fallback. Use when the launcher cannot guarantee
// the tile fits inside (M, N).
template <int ROWS, int COLS>
__device__ __forceinline__ void store_with_oob(
    float* __restrict__ C,
    const float (&sums)[ROWS][COLS],
    int rowBase, int colBase, int ty, int tx,
    int M, int N) {
    static_assert(COLS == 4, "store_with_oob specialised for COLS=4");

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = rowBase + ty * ROWS + r;
        if (row >= M) continue;
        const int col = colBase + tx * COLS;
        if (col + 3 < N) {
            const float4 out = { sums[r][0], sums[r][1], sums[r][2], sums[r][3] };
            *reinterpret_cast<float4*>(&C[row * N + col]) = out;
        } else {
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                if (col + c < N) C[row * N + col + c] = sums[r][c];
            }
        }
    }
}

// Supertile remap: turn a 1D block id into (bx, by) coordinates that walk
// SUPERxSUPER clusters of output tiles before moving to the next cluster
// (better L2 reuse than the linear column-major default). Returns false and
// leaves bx/by untouched if the remapped position is OOB.
template <int SUPER>
__device__ __forceinline__ bool supertile_remap(
    int linear_id, int blocks_x, int blocks_y,
    int& bx_out, int& by_out) {
    const int sxc = (blocks_x + SUPER - 1) / SUPER;
    const int sid = linear_id / (SUPER * SUPER);
    const int sub = linear_id % (SUPER * SUPER);
    const int sx = sid % sxc;
    const int sy = sid / sxc;
    const int subx = sub % SUPER;
    const int suby = sub / SUPER;
    const int bx_remap = sx * SUPER + subx;
    const int by_remap = sy * SUPER + suby;
    if (bx_remap >= blocks_x || by_remap >= blocks_y) return false;
    bx_out = bx_remap;
    by_out = by_remap;
    return true;
}

} // namespace gemm
