// shrink TILE_K from 64 -> 32 (tileA=64x32, tileB=32x64),
// keep TPB=4 to amortise the per-block work.
//
// Idea: SMEM per block drops 33 KB -> 16.6 KB. SMEM no longer binds the
// occupancy at 3 blocks/SM - it now allows up to 6 blocks/SM. Whether
// occupancy actually increases depends on the register limit, which is
// the second binding constraint. See ptxas output.
//
// Trade-off: TILE_K=32 means more numTiles per K-direction (K/32 vs K/64),
// so the K-tile load loop runs 2x more times. With TPB=4 each block also
// has 16 sub-tiles. Total syncs per block grows; need the higher occupancy
// to make up for it.
//
// Per-thread compute / register tile is unchanged: 8x4 = 32 sums.
// Per-thread load count is unchanged: 32 STS per K-iter (16 for tileA +
// 16 for tileB), even though each tile is now half the K-extent - because
// shrinking TILE_K halves both LOADS_A_PER_THREAD and LOADS_B_PER_THREAD.

#include "launchers.h"

#include <cassert>
#include <climits>


namespace {

constexpr int TILE_M          = 64;
constexpr int TILE_N          = 64;
constexpr int TILE_K          = 32;
constexpr int TILE_STRIDE_A   = TILE_K + 1;     // 33: pad+1 to break the 2-way LDS conflict (8ty group / 8ty+8 group hit same bank at stride 32).
constexpr int TILE_STRIDE_B   = TILE_N;         // 64: STS to tileB has 32 unique banks per warp, no pad needed.
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 8;
constexpr int ROWS_PER_THREAD = 8;
constexpr int COLS_PER_THREAD = 4;

// Hint at 4 blocks/SM. ptxas will cap regs at 65536 / (128 * 4) = 128/thread.
// gmem_vec used 168 regs/thread; the cap forces some compression. Whether
// this lands at 4 blocks/SM (32 % theoretical occupancy) or stays at 3
// (25 %) depends on what ptxas emits under the 128-reg cap.
template <int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 4)
void k32_tpb4_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float tileA[TILE_M][TILE_STRIDE_A];   // 64 x 33 x 4 = 8,448 B
    __shared__ float tileB[TILE_K][TILE_STRIDE_B];   // 32 x 64 x 4 = 8,192 B

    constexpr int NUM_THREADS        = BLOCK_DIM_X * BLOCK_DIM_Y;             // 128
    constexpr int LOADS_A_PER_THREAD = (TILE_M * TILE_K) / NUM_THREADS;       // 16
    constexpr int LOADS_B_PER_THREAD = (TILE_K * TILE_N) / NUM_THREADS;       // 16
    const int tid       = ty * BLOCK_DIM_X + tx;
    const int numTiles  = (K + TILE_K - 1) / TILE_K;

    for (int by = 0; by < TILES_PER_BLOCK; ++by) {
        for (int bx = 0; bx < TILES_PER_BLOCK; ++bx) {
            const int tileRowBase = (blockIdx.y * TILES_PER_BLOCK + by) * TILE_M;
            const int tileColBase = (blockIdx.x * TILES_PER_BLOCK + bx) * TILE_N;

            float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {{0.f}};

            for (int t = 0; t < numTiles; ++t) {
                const int tileOffset = t * TILE_K;

                // Load tileA: 64 rows x 32 cols = 2048 cells / 128 threads = 16/thread.
                // Per warp i=0: tid 0..31, idx 0..31. sRow = idx / 32 = 0 for tid 0..31
                //   (one warp covers one row). Bank = (33*sRow + sCol) % 32 = sCol % 32 for
                //   sRow=0 -> 32 unique banks.
                #pragma unroll
                for (int i = 0; i < LOADS_A_PER_THREAD; ++i) {
                    const int idx  = tid + i * NUM_THREADS;
                    const int sRow = idx / TILE_K;
                    const int sCol = idx % TILE_K;
                    const int aRow = tileRowBase + sRow;
                    const int aCol = tileOffset + sCol;
                    tileA[sRow][sCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.f;
                }

                // Load tileB: 32 rows x 64 cols = 2048 cells / 128 threads = 16/thread.
                // Per warp i=0: tid 0..31, idx 0..31. sRow = idx / 64 = 0, sCol = 0..31.
                // Bank = sCol % 32 -> 32 unique banks per warp.
                #pragma unroll
                for (int i = 0; i < LOADS_B_PER_THREAD; ++i) {
                    const int idx  = tid + i * NUM_THREADS;
                    const int sRow = idx / TILE_N;
                    const int sCol = idx % TILE_N;
                    const int bRow = tileOffset + sRow;
                    const int bCol = tileColBase + sCol;
                    tileB[sRow][sCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < TILE_K; ++k) {
                    float a[ROWS_PER_THREAD];
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r)
                        a[r] = tileA[ty * ROWS_PER_THREAD + r][k];

                    const float4 bv = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS_PER_THREAD]);
                    float b[COLS_PER_THREAD] = { bv.x, bv.y, bv.z, bv.w };

                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                        #pragma unroll
                        for (int c = 0; c < COLS_PER_THREAD; ++c)
                            sums[r][c] += a[r] * b[c];
                    }
                }

                __syncthreads();
            }

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int row = tileRowBase + ty * ROWS_PER_THREAD + r;
                if (row >= M) continue;
                const int col = tileColBase + tx * COLS_PER_THREAD;
                if (col + 3 < N) {
                    float4 out = { sums[r][0], sums[r][1], sums[r][2], sums[r][3] };
                    *reinterpret_cast<float4*>(&C[row * N + col]) = out;
                } else {
                    #pragma unroll
                    for (int c = 0; c < COLS_PER_THREAD; ++c) {
                        if (col + c < N)
                            C[row * N + col + c] = sums[r][c];
                    }
                }
            }
        }
    }
}

constexpr int MIN_BLOCKS_FOR_TPB4 = 64;
constexpr int MIN_BLOCKS_FOR_TPB2 = 128;

} // namespace

void k32_tpb4_launch(const GemmParams& p) {
    assert(static_cast<long long>(p.M) * p.N < static_cast<long long>(INT_MAX));

    if ((p.N & 3) != 0) {
        bank_pad_vec_launch(p);
        return;
    }

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    const int eff4   = TILE_M * 4;
    const int grid4x = (p.N + eff4 - 1) / eff4;
    const int grid4y = (p.M + eff4 - 1) / eff4;

    const int eff2   = TILE_M * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    if (grid4x * grid4y >= MIN_BLOCKS_FOR_TPB4) {
        dim3 grid(grid4x, grid4y);
        k32_tpb4_kernel<4><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y);
        k32_tpb4_kernel<2><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_M - 1) / TILE_M,
                  (p.M + TILE_M - 1) / TILE_M);
        k32_tpb4_kernel<1><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}
