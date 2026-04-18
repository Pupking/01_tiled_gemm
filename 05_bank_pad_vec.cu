// Layer 0 — v3: +1 col pad on tileA to break the 2-way row-stride LDS
// conflict warp_rebalance left on the table, plus LDS.128 on tileB.
//
// ptxas (sm_86, -O3): both <1> and <2> -> 168 regs, smem 33,024 B,
// 0 stack, 0 spill ld/st. 3 blocks/SM: 3 * 33,024 = 99,072 B <= 100 KB
// (static SMEM, no cudaFuncSetAttribute opt-in needed since 33 KB/block
// is under the 48 KB per-block static limit).
//
// Change vs warp_rebalance:
//   tileA stride 64 -> 65 (breaks 8-row bank conflict).
//   Per-thread cols: {tx, tx+16, tx+32, tx+48} -> {tx*4..tx*4+3}.
//   tileB b-load: 4x LDS.32 -> 1x LDS.128 (16B-aligned contiguous).
//   tileA a-load stays scalar (stride 65 is not 16B-aligned).

#include "gemm_common.h"

#include <cassert>
#include <climits>

namespace {

constexpr int TILE_SIZE       = 64;
constexpr int TILE_STRIDE_A   = TILE_SIZE + 1;   // pad+1: break 8-row LDS conflict. cp.async FORBIDDEN (§L0.1.2).
constexpr int TILE_STRIDE_B   = TILE_SIZE;       // no pad: contiguous-col b-load is already clean.
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 8;
constexpr int ROWS_PER_THREAD = 8;
constexpr int COLS_PER_THREAD = 4;               // 4 contiguous cols per thread: tx*4 .. tx*4+3

template <int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 3)
void bank_pad_vec_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float tileA[TILE_SIZE][TILE_STRIDE_A];
    __shared__ float tileB[TILE_SIZE][TILE_STRIDE_B];

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

                    // Single LDS.128: 4 contiguous floats at tx*4, 16B-aligned.
                    const float4 bv = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS_PER_THREAD]);
                    float b[COLS_PER_THREAD] = { bv.x, bv.y, bv.z, bv.w };

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
                    const int col = tileColBase + tx * COLS_PER_THREAD + cc;
                    if (col < N)
                        C[row * N + col] = sums[r][cc];
                }
            }
        }
    }
}

constexpr int MIN_BLOCKS_FOR_TPB2 = 128;

} // namespace

void bank_pad_vec_launch(const GemmParams& p) {
    assert(static_cast<long long>(p.M) * p.N < static_cast<long long>(INT_MAX));
    assert(static_cast<long long>(p.K)        < static_cast<long long>(INT_MAX));

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    const int eff2   = TILE_SIZE * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y);
        bank_pad_vec_kernel<2><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_SIZE - 1) / TILE_SIZE,
                  (p.M + TILE_SIZE - 1) / TILE_SIZE);
        bank_pad_vec_kernel<1><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}
