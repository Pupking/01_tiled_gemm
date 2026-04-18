// Layer 0 — v2: double warps/sched to cover the FMA dependency chain that
// starved v1.
//
// ptxas (sm_86, -O3): both <1> and <2> -> 168 regs, 32768 B smem,
// 0 stack, 0 spill ld/st. __launch_bounds__(128, 3) holds:
//   168 * 128 * 3 = 64,512 regs/SM  <= 65,536
//   3 * 32,768 B  = 98,304 B smem/SM <= 100 KB carveout.
//
// Change vs v1:
//   Block: 16x4 = 64 threads (2 warps) -> 16x8 = 128 threads (4 warps)
//   Per-thread reg block: 16x4 -> 8x4 accumulators (64 -> 32 FP32)
//   __launch_bounds__(128, 3) targets 3 resident blocks.
//   Theoretical occupancy 12.5% -> 25%.
//
// SMEM padding: pad-1 fixes the 2-way tileA ld conflict but bumps SMEM
// 32768 -> 33280 B, dropping 3 -> 2 blocks/SM at 100 KB carve-out. 3 blocks
// with residual conflicts beats 2 clean blocks here.

#include "gemm_common.h"

#include <cassert>
#include <climits>

namespace {

constexpr int TILE_SIZE       = 64;
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 8;
constexpr int ROWS_PER_THREAD = 8;                          // 8 * 8 = 64 rows
constexpr int COLS_PER_THREAD = TILE_SIZE / BLOCK_DIM_X;    // = 4 cols

template <int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 3)
void multi_tile_v2_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    constexpr int NUM_THREADS      = BLOCK_DIM_X * BLOCK_DIM_Y;                // 128
    constexpr int LOADS_PER_THREAD = (TILE_SIZE * TILE_SIZE) / NUM_THREADS;    // 32
    const int tid      = ty * BLOCK_DIM_X + tx;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int by = 0; by < TILES_PER_BLOCK; ++by) {
        for (int bx = 0; bx < TILES_PER_BLOCK; ++bx) {
            const int tileRowBase = (blockIdx.y * TILES_PER_BLOCK + by) * TILE_SIZE;
            const int tileColBase = (blockIdx.x * TILES_PER_BLOCK + bx) * TILE_SIZE;

            float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {{0.f}};

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
                    float a[ROWS_PER_THREAD];
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r)
                        a[r] = tileA[ty * ROWS_PER_THREAD + r][k];

                    float b[COLS_PER_THREAD];
                    #pragma unroll
                    for (int cc = 0; cc < COLS_PER_THREAD; ++cc)
                        b[cc] = tileB[k][cc * BLOCK_DIM_X + tx];

                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                        #pragma unroll
                        for (int cc = 0; cc < COLS_PER_THREAD; ++cc)
                            sums[r][cc] += a[r] * b[cc];
                    }
                }

                __syncthreads();
            }

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int row = tileRowBase + ty * ROWS_PER_THREAD + r;
                if (row >= M) continue;
                #pragma unroll
                for (int cc = 0; cc < COLS_PER_THREAD; ++cc) {
                    const int col = tileColBase + cc * BLOCK_DIM_X + tx;
                    if (col < N)
                        C[row * N + col] = sums[r][cc];
                }
            }
        }
    }
}

constexpr int MIN_BLOCKS_FOR_TPB2 = 128;

} // namespace

void warp_rebalance_launch (const GemmParams& p) {
    // §L0.1.4 — int offsets in the kernel; guard against overflow on large
    // shapes (unreachable on RTX 3050 memory budget, here for portability).
    assert(static_cast<long long>(p.M) * p.N < static_cast<long long>(INT_MAX));
    assert(static_cast<long long>(p.K)        < static_cast<long long>(INT_MAX));

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    const int eff2   = TILE_SIZE * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y);
        multi_tile_v2_kernel<2><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_SIZE - 1) / TILE_SIZE,
                  (p.M + TILE_SIZE - 1) / TILE_SIZE);
        multi_tile_v2_kernel<1><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}
