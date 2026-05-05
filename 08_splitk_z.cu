// Split-K parallelism via blockIdx.z. Spawns SPLIT_K parallel reductions
// over the K dimension; each block computes a K/SPLIT_K partial sum and
// atomicAdds into C.
//
// Motivation: at 512^3 the base kernel has only 64 blocks for 16 SMs and
// is under-saturated. SPLIT_K=N gives N x more blocks at no per-cell
// arithmetic cost, converting that idle SM time into useful work.
//
// Trade-off: atomicAdd on float is a serialised L2 RMW; SPLIT_K writers
// per cell bound the contention, but the cost is real. Wins when DRAM/L2
// has slack AND the grid is small. Up-front costs:
//   1. cudaMemsetAsync(C, 0) so atomicAdd starts from a known state
//      (the bench harness poisons C before each launch).
//   2. C-store path drops STG.128; atomicAdd is scalar.

#include "gemm_common.h"

#include <cassert>
#include <climits>

namespace {

constexpr int TILE_SIZE       = 64;
constexpr int TILE_STRIDE_A   = TILE_SIZE + 1;
constexpr int TILE_STRIDE_B   = TILE_SIZE;
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 8;
constexpr int ROWS_PER_THREAD = 8;
constexpr int COLS_PER_THREAD = 4;
constexpr int MIN_BLOCKS_FOR_TPB4 = 64;
constexpr int MIN_BLOCKS_FOR_TPB2 = 128;

template <int SPLIT_K, int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 3)
void splitk_kernel(const float* __restrict__ A,
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

    // Per-blockIdx.z K range. K is partitioned into SPLIT_K contiguous
    // chunks; each block processes its own chunk and atomically adds into C.
    const int kPerSplit = (K + SPLIT_K - 1) / SPLIT_K;
    const int kStart    = blockIdx.z * kPerSplit;
    const int kEnd      = (kStart + kPerSplit < K) ? (kStart + kPerSplit) : K;
    const int kLen      = kEnd - kStart;
    const int numTiles  = (kLen + TILE_SIZE - 1) / TILE_SIZE;

    for (int by = 0; by < TILES_PER_BLOCK; ++by) {
        for (int bx = 0; bx < TILES_PER_BLOCK; ++bx) {
            const int tileRowBase = (blockIdx.y * TILES_PER_BLOCK + by) * TILE_SIZE;
            const int tileColBase = (blockIdx.x * TILES_PER_BLOCK + bx) * TILE_SIZE;

            float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {{0.f}};

            for (int t = 0; t < numTiles; ++t) {
                const int tileOffset = kStart + t * TILE_SIZE;

                #pragma unroll
                for (int i = 0; i < LOADS_PER_THREAD; ++i) {
                    const int idx  = tid + i * NUM_THREADS;
                    const int sRow = idx / TILE_SIZE;
                    const int sCol = idx % TILE_SIZE;

                    const int aRow = tileRowBase + sRow;
                    const int aCol = tileOffset + sCol;
                    tileA[sRow][sCol] = (aRow < M && aCol < kEnd) ? A[aRow * K + aCol] : 0.f;

                    const int bRow = tileOffset + sRow;
                    const int bCol = tileColBase + sCol;
                    tileB[sRow][sCol] = (bRow < kEnd && bCol < N) ? B[bRow * N + bCol] : 0.f;
                }

                __syncthreads();

                #pragma unroll
                for (int k = 0; k < TILE_SIZE; ++k) {
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
                #pragma unroll
                for (int c = 0; c < COLS_PER_THREAD; ++c) {
                    if (col + c < N) {
                        if constexpr (SPLIT_K == 1) {
                            C[row * N + col + c] = sums[r][c];
                        } else {
                            atomicAdd(&C[row * N + col + c], sums[r][c]);
                        }
                    }
                }
            }
        }
    }
}

template <int SPLIT_K>
void splitk_launch_t(const GemmParams& p) {
    assert(static_cast<long long>(p.M) * p.N < static_cast<long long>(INT_MAX));

    if constexpr (SPLIT_K > 1) {
        // Init C=0 so atomicAdd accumulates from a known state. The bench
        // harness poisons C; we need to overwrite the poison.
        CUDA_CHECK(cudaMemsetAsync(p.dC, 0, static_cast<size_t>(p.M) * p.N * sizeof(float)));
    }

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    const int eff4   = TILE_SIZE * 4;
    const int grid4x = (p.N + eff4 - 1) / eff4;
    const int grid4y = (p.M + eff4 - 1) / eff4;

    const int eff2   = TILE_SIZE * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    // Same TPB heuristic as tpb4 but accounting for the SPLIT_K extra grid
    // dim - keep a sensible total block count.
    if (grid4x * grid4y * SPLIT_K >= MIN_BLOCKS_FOR_TPB4) {
        dim3 grid(grid4x, grid4y, SPLIT_K);
        splitk_kernel<SPLIT_K, 4><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else if (grid2x * grid2y * SPLIT_K >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y, SPLIT_K);
        splitk_kernel<SPLIT_K, 2><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_SIZE - 1) / TILE_SIZE,
                  (p.M + TILE_SIZE - 1) / TILE_SIZE,
                  SPLIT_K);
        splitk_kernel<SPLIT_K, 1><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}

} // namespace

void splitk1_launch(const GemmParams& p) { splitk_launch_t<1>(p); }
void splitk2_launch(const GemmParams& p) { splitk_launch_t<2>(p); }
void splitk3_launch(const GemmParams& p) { splitk_launch_t<3>(p); }
void splitk4_launch(const GemmParams& p) { splitk_launch_t<4>(p); }
