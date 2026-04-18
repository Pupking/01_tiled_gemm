// Layer 0 — Coalesced writes via column-aligned thread mapping.
//
// Correctness note:
//   BLOCK_DIM_X (16) < TILE_SIZE (32), so each thread writes a 2-column
//   stripe of the tile; BLOCK_DIM_Y * ROWS_PER_THREAD (4*8=32) covers all
//   32 rows of the tile. A prior version had ROWS_PER_THREAD=4, which
//   silently wrote only the top 16 rows × left 16 cols of each output tile
//   — a 25%-coverage bug that verify happened to miss because the
//   un-overwritten cells still held the cuBLAS reference output from the
//   prior benchmark run. Current verify pass now poisons the output buffer
//   before every launch (common/bench_harness.h::poison_output) so that
//   class of bug can't repeat.
//
// Block geometry:  16 × 4 = 64 threads (2 warps per block)
// Per-thread work: ROWS_PER_THREAD × COLS_PER_THREAD = 8 × 2 = 16
//                  FP32 accumulators in registers.
//
// Store coalescing: within a warp, tx varies 0..15 with (ty, r, cc) uniform.
//   - Each half-warp writes 16 consecutive 4-byte floats = 64 B = 2 sectors,
//     fully utilised.
//   - The two halves of a warp hit two different rows (ty=0 vs ty=1 within
//     the warp), so each warp produces 4 sector-aligned writes per output
//     row, not one wide one. This is "coalesced" at the sector level — the
//     hardware sees no wasted bytes per transaction.

#include "gemm_common.h"

namespace {

constexpr int TILE_SIZE       = 32;
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 4;
constexpr int ROWS_PER_THREAD = 8;                          // covers 4 * 8 = 32 rows
constexpr int COLS_PER_THREAD = TILE_SIZE / BLOCK_DIM_X;    // = 2 cols per thread

__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y)
void tiled_coalesced_gemm_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    const int tx = threadIdx.x;   // 0..15
    const int ty = threadIdx.y;   // 0..3

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {{0.f}};

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int tileOffset = t * TILE_SIZE;

        constexpr int NUM_THREADS      = BLOCK_DIM_X * BLOCK_DIM_Y;   // 64
        constexpr int LOADS_PER_THREAD = (TILE_SIZE * TILE_SIZE) / NUM_THREADS;  // 16
        const int tid = ty * BLOCK_DIM_X + tx;

        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; ++i) {
            const int idx  = tid + i * NUM_THREADS;
            const int sRow = idx / TILE_SIZE;
            const int sCol = idx % TILE_SIZE;

            const int aRow = blockIdx.y * TILE_SIZE + sRow;
            const int aCol = tileOffset + sCol;
            tileA[sRow][sCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.f;

            const int bRow = tileOffset + sRow;
            const int bCol = blockIdx.x * TILE_SIZE + sCol;
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
        const int row = blockIdx.y * TILE_SIZE + ty * ROWS_PER_THREAD + r;
        if (row >= M) continue;
        #pragma unroll
        for (int cc = 0; cc < COLS_PER_THREAD; ++cc) {
            const int col = blockIdx.x * TILE_SIZE + cc * BLOCK_DIM_X + tx;
            if (col < N)
                C[row * N + col] = sums[r][cc];
        }
    }
}

} // namespace

void tiled_coalesced_launch(const GemmParams& p) {
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((p.N + TILE_SIZE - 1) / TILE_SIZE,
              (p.M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_coalesced_gemm_kernel<<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    CUDA_CHECK_LAST();
}
