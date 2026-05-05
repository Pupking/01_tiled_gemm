// Coalesced writes via column-aligned thread mapping.
//
// Block:        16 x 4 = 64 threads (2 warps per block).
// Per-thread:   ROWS_PER_THREAD x COLS_PER_THREAD = 8 x 2 = 16 FP32 sums.
// Coverage:     BLOCK_DIM_Y * ROWS_PER_THREAD = 4*8 = 32 = TILE_SIZE; each
//               thread writes a 2-column stripe spanning the full 32 rows.
//               (A prior version with ROWS_PER_THREAD=4 silently wrote 25%
//               of each tile - poison_output in the harness now catches that
//               class of bug at verify time.)
// Store path:   tx=0..15 within a warp; each half-warp writes 16 consecutive
//               floats = 64 B = 2 sectors, fully utilised. "Coalesced" at the
//               sector level (no wasted bytes per transaction) even though
//               there are 4 sector writes per output row, not one wide STG.

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
            for (int c = 0; c < COLS_PER_THREAD; ++c)
                b[c] = tileB[k][c * BLOCK_DIM_X + tx];

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
        const int row = blockIdx.y * TILE_SIZE + ty * ROWS_PER_THREAD + r;
        if (row >= M) continue;
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
            const int col = blockIdx.x * TILE_SIZE + c * BLOCK_DIM_X + tx;
            if (col < N)
                C[row * N + col] = sums[r][c];
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
