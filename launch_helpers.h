#pragma once

// Reusable launch dispatchers for the wider_m / supertile / ldg_* family.
// Each dispatcher receives the two kernel instantiations (TPB=1 and TPB=2)
// as function pointers, opts in to dynamic SMEM, and picks one based on
// shape divisibility and grid size.
//
// Older kernels (07, 08, 11, 12, 13, 14) use a different TPB=4 dispatch and
// pass M/N/K directly to the kernel; they don't fit this contract.

#include "gemm_common.h"

namespace launch {

constexpr int MIN_BLOCKS_FOR_TPB2 = 64;

using Kernel2D = void (*)(const float*, const float*, float*, int, int, int);
using Kernel1D = void (*)(const float*, const float*, float*, int, int, int, int, int);

// 2D grid dispatch. Both kernels take (A, B, C, M, N, K). TPB=2 packs a 2x2
// block of TILE_MxTILE_N output tiles per block; falls back to TPB=1 when
// shape isn't divisible by 2*TILE or grid would shrink below MIN_BLOCKS_FOR_TPB2.
inline void grid2d_tpb(const GemmParams& p, dim3 block, int smem_bytes,
                       int tile_m, int tile_n,
                       Kernel2D kernel_tpb1, Kernel2D kernel_tpb2) {
    enable_dynamic_smem(kernel_tpb1, smem_bytes);
    enable_dynamic_smem(kernel_tpb2, smem_bytes);

    const int eff2_m = tile_m * 2;
    const int eff2_n = tile_n * 2;
    const int g2x = (p.N + eff2_n - 1) / eff2_n;
    const int g2y = (p.M + eff2_m - 1) / eff2_m;
    const bool can_tpb2 = (p.M % eff2_m == 0) && (p.N % eff2_n == 0);

    if (can_tpb2 && g2x * g2y >= MIN_BLOCKS_FOR_TPB2) {
        kernel_tpb2<<<dim3(g2x, g2y), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        const int g1x = (p.N + tile_n - 1) / tile_n;
        const int g1y = (p.M + tile_m - 1) / tile_m;
        kernel_tpb1<<<dim3(g1x, g1y), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}

// 1D-grid supertile dispatch. Kernels take (A, B, C, M, N, K, blocks_x,
// blocks_y) and remap blockIdx.x via a supertile walk for L2 reuse.
inline void grid1d_supertile_tpb(const GemmParams& p, dim3 block, int smem_bytes,
                                 int tile_m, int tile_n,
                                 Kernel1D kernel_tpb1, Kernel1D kernel_tpb2) {
    enable_dynamic_smem(kernel_tpb1, smem_bytes);
    enable_dynamic_smem(kernel_tpb2, smem_bytes);

    const int eff2_m = tile_m * 2;
    const int eff2_n = tile_n * 2;
    const int g2x = (p.N + eff2_n - 1) / eff2_n;
    const int g2y = (p.M + eff2_m - 1) / eff2_m;
    const bool can_tpb2 = (p.M % eff2_m == 0) && (p.N % eff2_n == 0);

    if (can_tpb2 && g2x * g2y >= MIN_BLOCKS_FOR_TPB2) {
        kernel_tpb2<<<dim3(g2x * g2y, 1), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g2x, g2y);
    } else {
        const int g1x = (p.N + tile_n - 1) / tile_n;
        const int g1y = (p.M + tile_m - 1) / tile_m;
        kernel_tpb1<<<dim3(g1x * g1y, 1), block, smem_bytes>>>(
            p.dA, p.dB, p.dC, p.M, p.N, p.K, g1x, g1y);
    }
    CUDA_CHECK_LAST();
}

// 1D-grid supertile, TPB=1 only. The launch grid must cover the *padded*
// supertile space (sxc*syc*SUPER^2) so that every (bx, by) the supertile remap
// can produce gets a block; OOB positions return early inside the kernel.
inline void grid1d_supertile_padded(const GemmParams& p, dim3 block, int smem_bytes,
                                    int tile_m, int tile_n, int super,
                                    Kernel1D kernel) {
    enable_dynamic_smem(kernel, smem_bytes);

    const int g1x = (p.N + tile_n - 1) / tile_n;
    const int g1y = (p.M + tile_m - 1) / tile_m;
    const int sxc = (g1x + super - 1) / super;
    const int syc = (g1y + super - 1) / super;
    const int total = sxc * syc * super * super;

    kernel<<<dim3(total, 1), block, smem_bytes>>>(
        p.dA, p.dB, p.dC, p.M, p.N, p.K, g1x, g1y);
    CUDA_CHECK_LAST();
}

} // namespace launch
