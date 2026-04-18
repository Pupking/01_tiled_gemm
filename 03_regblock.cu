// Layer 0 — 64×64 tiles, each block sweeps a 1×1 or 2×2 region of output
// tiles (TILES_PER_BLOCK template param, picked adaptively at launch).
//
// ptxas (sm_86, -O3): regs=___, smem=32768 B (fill from build log).
//
// Correctness note:
//   BLOCK_DIM_X (16) < TILE_SIZE (64), so each thread writes a 4-column
//   stripe of the tile; BLOCK_DIM_Y * ROWS_PER_THREAD (4*16=64) covers all
//   64 rows. A prior version had ROWS_PER_THREAD=8, which silently wrote
//   only 32 rows × 16 cols of the 64×64 tile — 12.5% coverage. Verify
//   passed because the un-overwritten cells still held cuBLAS output from
//   the previous run. common/bench_harness.h::poison_output in the harness
//   now prevents that class of bug from recurring.
//
// Block geometry:  16 × 4 = 64 threads (2 warps per block)
// Per-thread work: ROWS_PER_THREAD × COLS_PER_THREAD = 16 × 4 = 64
//                  FP32 accumulators in registers. Tight but under SM86's
//                  255-reg/thread cap; SMEM (32 KB) is the binding
//                  occupancy constraint, not registers.
//
// SMEM per block: 2 × 64 × 64 × 4 B = 32 KB.
//   - Fits under the 48 KB static per-block SMEM limit (no carve-out).
//   - 3 blocks/SM at 100 KB per-SM cap → low-occupancy / high-ILP regime.

#include "gemm_common.h"

#include <cassert>
#include <climits>

namespace {

constexpr int TILE_SIZE       = 64;
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 4;
constexpr int ROWS_PER_THREAD = 16;                         // covers 4 * 16 = 64 rows
constexpr int COLS_PER_THREAD = TILE_SIZE / BLOCK_DIM_X;    // = 4 cols per thread

template <int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y)
void multi_tile_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    constexpr int NUM_THREADS      = BLOCK_DIM_X * BLOCK_DIM_Y;
    constexpr int LOADS_PER_THREAD = (TILE_SIZE * TILE_SIZE) / NUM_THREADS;  // 64
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

// Adaptive launch: 2×2 tiles-per-block only when the resulting grid still
// saturates SMs; otherwise fall back to 1×1.
constexpr int MIN_BLOCKS_FOR_TPB2 = 128;

} // namespace

void regblock_launch(const GemmParams& p) {
    // §L0.1.4 — tileRowBase/tileColBase are `int`. They wrap once
    // blockIdx.{x,y} * TPB * TILE_SIZE exceeds INT_MAX. Runtime-check the
    // cheap shape bound (audit's alternative to promoting every offset to
    // int64_t). RTX 3050 memory cap makes this unreachable on-device;
    // the assert is here for portability to larger GPUs.
    assert(static_cast<long long>(p.M) * p.N < static_cast<long long>(INT_MAX));
    assert(static_cast<long long>(p.K)        < static_cast<long long>(INT_MAX));

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    const int eff2   = TILE_SIZE * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y);
        multi_tile_kernel<2><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_SIZE - 1) / TILE_SIZE,
                  (p.M + TILE_SIZE - 1) / TILE_SIZE);
        multi_tile_kernel<1><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}
