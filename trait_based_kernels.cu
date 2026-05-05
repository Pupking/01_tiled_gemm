// Trait-based kernel family. Twelve registry entries implemented as
// (Traits, launcher) pairs over the shared tiled_kernel template
// (tiled_kernel.h). Each block below is one optimization step; the
// delta versus the previous step is in the Traits fields and the
// launcher choice.
//
// Order matches the optimization journey, not file numbering.

#include "launchers.h"
#include "tiled_kernel.h"

namespace {

// Base 64x64 tile, 128 threads, scalar OOB loads, STG.128 store via
// store_with_oob (selected when Load::CLEAN==false). Predecessor
// bank_pad_vec used 4x STG.32 per row at 8/32 bytes/sector; STG.128
// raises sector utilization to 32/32.
struct GmemVecTraits {
    static constexpr int TILE_M = 64, TILE_N = 64, TILE_K = 64;
    static constexpr int STRIDE_A = TILE_K + 1;
    static constexpr int STRIDE_B = TILE_N;
    static constexpr int BX = 16, BY = 8;
    static constexpr int ROWS = 8, COLS = 4;
    static constexpr int LB_BPS = 3;
    using Load  = tiled::ScalarOOB;
    using Remap = tiled::Linear;
};

// Same kernel as gmem_vec; launcher tries TPB=4 first (4x4 sub-tiles
// per block, 256x256 effective output region) before falling back to
// TPB=2/1. Each unique tile-load is now reused 4x via L1/L2 instead of 2x.
using Tpb4Traits = GmemVecTraits;

// 128x64 tile, 256 threads, 8x4 reg block. Per-cell global traffic
// drops from (1/64+1/64) to (1/128+1/64), about 30% less DRAM. SMEM
// 49.7 KB/block, fits 2 blocks/SM (33% occupancy vs gmem_vec's 25%).
struct WiderMTraits {
    static constexpr int TILE_M = 128, TILE_N = 64, TILE_K = 64;
    static constexpr int STRIDE_A = TILE_K + 1;
    static constexpr int STRIDE_B = TILE_N;
    static constexpr int BX = 16, BY = 16;
    static constexpr int ROWS = 8, COLS = 4;
    static constexpr int LB_BPS = 2;
    using Load  = tiled::ScalarOOB;
    using Remap = tiled::Linear;
};

// wider_m + 4x4 supertile remap. Consecutive blocks land in the same
// L2-friendly cluster instead of column-major order.
struct SupertileTraits : WiderMTraits { using Remap = tiled::Supertile<4>; };

// wider_m with OOB checks dropped. Launcher gates on full divisibility
// so the always-true branches go away entirely.
struct DropOobTraits  : WiderMTraits {
    using Load  = tiled::ScalarClean;
    using Remap = tiled::Linear;
};

// wider_m + 8x8 supertile (independent gain stacks with the wider tile).
struct SuperWiderTraits : WiderMTraits { using Remap = tiled::Supertile<8>; };

// wider_m geometry with float4 LDG.128 cooperative loads. Cuts LDG
// instruction count 4x and frees scheduler slots for FMAs.
struct LdgVecTraits   : WiderMTraits { using Load = tiled::Vec4Clean; };

// ldg_vec + supertile-8.
struct LdgSuperTraits    : LdgVecTraits { using Remap = tiled::Supertile<8>; };

// ldg_super with a wider supertile cluster (16 vs 8).
struct LdgSuperBigTraits : LdgVecTraits { using Remap = tiled::Supertile<16>; };

// ldg_super with TILE_K=128 (deeper K loop, fewer outer iterations).
// SMEM 98.8 KB, just under the 99 KB sm_86 carveout, 1 block/SM.
struct LdgK128Traits {
    static constexpr int TILE_M = 128, TILE_N = 64, TILE_K = 128;
    static constexpr int STRIDE_A = TILE_K + 1;
    static constexpr int STRIDE_B = TILE_N;
    static constexpr int BX = 16, BY = 16;
    static constexpr int ROWS = 8, COLS = 4;
    static constexpr int LB_BPS = 1;
    using Load  = tiled::Vec4Clean;
    using Remap = tiled::Supertile<8>;
};

// 256x64 tile, 64 sums/thread - large reg footprint, watch for spills.
// Padded supertile launcher to guarantee full coverage at small grids.
struct LdgWider2Traits {
    static constexpr int TILE_M = 256, TILE_N = 64, TILE_K = 64;
    static constexpr int STRIDE_A = TILE_K + 1;
    static constexpr int STRIDE_B = TILE_N;
    static constexpr int BX = 16, BY = 16;
    static constexpr int ROWS = 16, COLS = 4;
    static constexpr int LB_BPS = 1;
    using Load  = tiled::Vec4Clean;
    using Remap = tiled::Supertile<8>;
};

// 256x128 tile, 512 threads (16 warps/SM). SMEM 99.3 KB, 1 block/SM.
struct LdgMaxTraits {
    static constexpr int TILE_M = 256, TILE_N = 128, TILE_K = 64;
    static constexpr int STRIDE_A = TILE_K + 1;
    static constexpr int STRIDE_B = TILE_N;
    static constexpr int BX = 32, BY = 16;
    static constexpr int ROWS = 16, COLS = 4;
    static constexpr int LB_BPS = 1;
    using Load  = tiled::Vec4Clean;
    using Remap = tiled::Supertile<8>;
};

// Vec4 loads need N divisible by 4 *and* the tile to fit cleanly.
// Scalar OOB loads only need N % 4 == 0 (for the STG.128 path).
template <class T>
inline bool needs_clean_shape(const GemmParams& p) {
    return (p.M % T::TILE_M) || (p.N % T::TILE_N) ||
           (p.K % T::TILE_K) || (p.N & 3);
}

inline bool needs_n_aligned(const GemmParams& p) {
    return (p.N & 3) != 0;
}

} // namespace

void gmem_vec_launch     (const GemmParams& p) { if (needs_n_aligned(p))             { bank_pad_vec_launch(p); return; } tiled::launch_2d              <GmemVecTraits     >(p); }
void tpb4_launch         (const GemmParams& p) { if (needs_n_aligned(p))             { bank_pad_vec_launch(p); return; } tiled::launch_2d_tpb4         <Tpb4Traits        >(p); }
void wider_m_launch      (const GemmParams& p) { if (needs_n_aligned(p))             { bank_pad_vec_launch(p); return; } tiled::launch_2d              <WiderMTraits      >(p); }
void supertile_launch    (const GemmParams& p) { if (needs_n_aligned(p))             { bank_pad_vec_launch(p); return; } tiled::launch_supertile       <SupertileTraits   >(p); }
void super_wider_launch  (const GemmParams& p) { if (needs_n_aligned(p))             { bank_pad_vec_launch(p); return; } tiled::launch_supertile       <SuperWiderTraits  >(p); }
void drop_oob_launch     (const GemmParams& p) { if (needs_clean_shape<DropOobTraits  >(p)) { bank_pad_vec_launch(p); return; } tiled::launch_2d        <DropOobTraits     >(p); }
void ldg_vec_launch      (const GemmParams& p) { if (needs_clean_shape<LdgVecTraits   >(p)) { bank_pad_vec_launch(p); return; } tiled::launch_2d        <LdgVecTraits      >(p); }
void ldg_super_launch    (const GemmParams& p) { if (needs_clean_shape<LdgSuperTraits >(p)) { bank_pad_vec_launch(p); return; } tiled::launch_supertile <LdgSuperTraits    >(p); }
void ldg_super_big_launch(const GemmParams& p) { if (needs_clean_shape<LdgSuperBigTraits>(p)){ bank_pad_vec_launch(p); return; } tiled::launch_supertile<LdgSuperBigTraits >(p); }
void ldg_k128_launch     (const GemmParams& p) { if (needs_clean_shape<LdgK128Traits  >(p)) { bank_pad_vec_launch(p); return; } tiled::launch_supertile <LdgK128Traits     >(p); }
void ldg_wider2_launch   (const GemmParams& p) { if (needs_clean_shape<LdgWider2Traits>(p)) { bank_pad_vec_launch(p); return; } tiled::launch_supertile_padded<LdgWider2Traits>(p); }
void ldg_max_launch      (const GemmParams& p) { if (needs_clean_shape<LdgMaxTraits   >(p)) { bank_pad_vec_launch(p); return; } tiled::launch_supertile_padded<LdgMaxTraits   >(p); }
