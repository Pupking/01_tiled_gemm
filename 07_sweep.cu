// Parametric sweep of block geometry on the tpb4 base.
//
// Goto & van de Geijn ("Anatomy of High-Performance Matrix Multiplication",
// ACM TOMS 2008) frames GEMM as three nested blockings: cache (mc, kc, nc),
// TLB-aware, and register (mr, nr). Per-thread arithmetic intensity is
//   I_reg = (mr * nr) / (mr + nr)
// each iteration loads (mr+nr) operands for (mr*nr) FMAs, so larger
// I_reg = more compute per LDS. CPU GEMM targets mr=nr=4..8, I_reg=2..4.
//
// The CPU mc/kc/nc cache blocking maps to our SMEM tile (TILE_SIZE=64 in
// M/N, TILE_K=64); (mr, nr) maps to ROWS_PER_THREAD x COLS_PER_THREAD;
// the CPU "panel packing" maps to our cooperative SMEM load.
//
// Caveat for GPU: bigger reg-block raises per-thread register pressure ->
// fewer resident blocks/SM -> less latency hiding. CPU has no analogue
// (OoO + SMT do that implicitly). Pick (BX, BY, ROWS, COLS) under the
// constraint BX*COLS = BY*ROWS = TILE_SIZE and find the wall-clock optimum
// empirically. Cache blocking is held fixed from tpb4 (TILE=64, TPB=4).
//
//   Config | BLOCK_DIM | reg tile | threads | warps | sums | I_reg
//   -------|-----------|----------|---------|-------|------|------
//   A      | (16, 8)   | 8 x 4    | 128     | 4     | 32   | 2.67  <- baseline (gmem_vec/tpb4)
//   B      | (16, 16)  | 4 x 4    | 256     | 8     | 16   | 2.00
//   C      | ( 8, 16)  | 4 x 8    | 128     | 4     | 32   | 2.67  <- transpose of A
//   D      | ( 8,  8)  | 8 x 8    | 64      | 2     | 64   | 4.00  <- Goto-optimal I_reg
//   E      | (32, 16)  | 4 x 2    | 512     | 16    | 8    | 1.33  <- max warps/block
//   F      | (32,  8)  | 8 x 2    | 256     | 8     | 16   | 1.60

#include "launchers.h"

#include <cassert>
#include <climits>


namespace {

constexpr int TILE_SIZE        = 64;
constexpr int TILE_STRIDE_A    = TILE_SIZE + 1;
constexpr int TILE_STRIDE_B    = TILE_SIZE;
constexpr int MIN_BLOCKS_FOR_TPB4 = 64;
constexpr int MIN_BLOCKS_FOR_TPB2 = 128;

template <int BX, int BY, int ROWS, int COLS, int LB_MIN_BLOCKS, int TILES_PER_BLOCK>
__global__ __launch_bounds__(BX * BY, LB_MIN_BLOCKS)
void sweep_kernel(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float* __restrict__ C,
                  int M, int N, int K) {
    static_assert(BX * COLS == TILE_SIZE, "BLOCK_DIM_X * COLS must equal TILE_SIZE");
    static_assert(BY * ROWS == TILE_SIZE, "BLOCK_DIM_Y * ROWS must equal TILE_SIZE");
    static_assert((TILE_SIZE * TILE_SIZE) % (BX * BY) == 0, "tile must be divisible by NUM_THREADS");

    constexpr int NUM_THREADS      = BX * BY;
    constexpr int LOADS_PER_THREAD = (TILE_SIZE * TILE_SIZE) / NUM_THREADS;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * BX + tx;

    __shared__ float tileA[TILE_SIZE][TILE_STRIDE_A];
    __shared__ float tileB[TILE_SIZE][TILE_STRIDE_B];

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int by = 0; by < TILES_PER_BLOCK; ++by) {
        for (int bx = 0; bx < TILES_PER_BLOCK; ++bx) {
            const int tileRowBase = (blockIdx.y * TILES_PER_BLOCK + by) * TILE_SIZE;
            const int tileColBase = (blockIdx.x * TILES_PER_BLOCK + bx) * TILE_SIZE;

            float sums[ROWS][COLS] = {{0.f}};

            for (int t = 0; t < numTiles; ++t) {
                const int tileOffset = t * TILE_SIZE;

                #pragma unroll
                for (int i = 0; i < LOADS_PER_THREAD; ++i) {
                    const int idx  = tid + i * NUM_THREADS;
                    const int sRow = idx / TILE_SIZE;
                    const int sCol = idx % TILE_SIZE;

                    const int aRow = tileRowBase + sRow;
                    const int aCol = tileOffset + sCol;
                    tileA[sRow][sCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.f;

                    const int bRow = tileOffset + sRow;
                    const int bCol = tileColBase + sCol;
                    tileB[sRow][sCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < TILE_SIZE; ++k) {
                    float a[ROWS];
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r)
                        a[r] = tileA[ty * ROWS + r][k];

                    float b[COLS];
                    // Vectorize tileB load when COLS is a multiple of 4 and the
                    // base is 16-byte aligned (tx*COLS*4 must be 16-aligned).
                    if constexpr (COLS == 8) {
                        const float4 bv0 = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS]);
                        const float4 bv1 = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS + 4]);
                        b[0] = bv0.x; b[1] = bv0.y; b[2] = bv0.z; b[3] = bv0.w;
                        b[4] = bv1.x; b[5] = bv1.y; b[6] = bv1.z; b[7] = bv1.w;
                    } else if constexpr (COLS == 4) {
                        const float4 bv = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS]);
                        b[0] = bv.x; b[1] = bv.y; b[2] = bv.z; b[3] = bv.w;
                    } else if constexpr (COLS == 2) {
                        const float2 bv = *reinterpret_cast<const float2*>(&tileB[k][tx * COLS]);
                        b[0] = bv.x; b[1] = bv.y;
                    } else {
                        #pragma unroll
                        for (int c = 0; c < COLS; ++c)
                            b[c] = tileB[k][tx * COLS + c];
                    }

                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        #pragma unroll
                        for (int c = 0; c < COLS; ++c)
                            sums[r][c] += a[r] * b[c];
                    }
                }

                __syncthreads();
            }

            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int row = tileRowBase + ty * ROWS + r;
                if (row >= M) continue;
                const int col = tileColBase + tx * COLS;
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    if (col + c < N)
                        C[row * N + col + c] = sums[r][c];
                }
            }
        }
    }
}

template <int BX, int BY, int ROWS, int COLS, int LB>
void launch_sweep(const GemmParams& p) {
    dim3 block(BX, BY);

    const int eff4   = TILE_SIZE * 4;
    const int grid4x = (p.N + eff4 - 1) / eff4;
    const int grid4y = (p.M + eff4 - 1) / eff4;

    const int eff2   = TILE_SIZE * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    if (grid4x * grid4y >= MIN_BLOCKS_FOR_TPB4) {
        dim3 grid(grid4x, grid4y);
        sweep_kernel<BX, BY, ROWS, COLS, LB, 4><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y);
        sweep_kernel<BX, BY, ROWS, COLS, LB, 2><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_SIZE - 1) / TILE_SIZE,
                  (p.M + TILE_SIZE - 1) / TILE_SIZE);
        sweep_kernel<BX, BY, ROWS, COLS, LB, 1><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}

} // namespace

// LB_MIN_BLOCKS hint chosen per config by counting reg pressure and the
// SMEM-derived ceiling of 3 blocks/SM at TILE=64.
//   A,B,C,F: ptxas naturally lands <=168 regs -> 3 blocks/SM possible
//   D: 8x8 reg tile = 64 sums -> very high reg pressure; allow only 1 block
//   E: 512 threads x any regs >= 64 -> only 2 blocks possible by reg, set 2
void sweep_A_launch(const GemmParams& p) { launch_sweep<16,  8, 8, 4, 3>(p); }
void sweep_B_launch(const GemmParams& p) { launch_sweep<16, 16, 4, 4, 3>(p); }
void sweep_C_launch(const GemmParams& p) { launch_sweep< 8, 16, 4, 8, 3>(p); }
void sweep_D_launch(const GemmParams& p) { launch_sweep< 8,  8, 8, 8, 1>(p); }
void sweep_E_launch(const GemmParams& p) { launch_sweep<32, 16, 4, 2, 1>(p); }
void sweep_F_launch(const GemmParams& p) { launch_sweep<32,  8, 8, 2, 2>(p); }
