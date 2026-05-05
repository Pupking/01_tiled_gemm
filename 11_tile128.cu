// 128x128 output tile, 512 threads, 4x8 reg block. Matches cuBLAS's
// per-block geometry. Per-cell global traffic drops from 3/256 (wider_m's
// 128x64) to 2/256 here - 33 % less DRAM per cell.
//
// Block:      BLOCK_DIM = (16, 32) = 512 threads = 16 warps.
// Per-thread: ROWS x COLS = 4 x 8 = 32 sums (same footprint as gmem_vec).
// SMEM/block: 128*65*4 + 64*128*4 = 66,048 B; SMEM-bound to 1 block/SM,
//             16 warps/SM = 33 % occupancy.

#include "launchers.h"

#include <cassert>
#include <climits>


namespace {

constexpr int TILE_M          = 128;
constexpr int TILE_N          = 128;
constexpr int TILE_K          = 64;
constexpr int TILE_STRIDE_A   = TILE_K + 1;
constexpr int TILE_STRIDE_B   = TILE_N;
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 32;
constexpr int ROWS_PER_THREAD = 4;
constexpr int COLS_PER_THREAD = 8;

constexpr int TILE_A_ELEMS = TILE_M * TILE_STRIDE_A;
constexpr int TILE_B_ELEMS = TILE_K * TILE_STRIDE_B;

template <int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 1)
void tile128_kernel(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    extern __shared__ float smem[];
    float (*tileA)[TILE_STRIDE_A] = reinterpret_cast<float (*)[TILE_STRIDE_A]>(smem);
    float (*tileB)[TILE_STRIDE_B] = reinterpret_cast<float (*)[TILE_STRIDE_B]>(smem + TILE_A_ELEMS);

    constexpr int NUM_THREADS        = BLOCK_DIM_X * BLOCK_DIM_Y;             // 512
    constexpr int LOADS_A_PER_THREAD = (TILE_M * TILE_K) / NUM_THREADS;       // 128*64/512 = 16
    constexpr int LOADS_B_PER_THREAD = (TILE_K * TILE_N) / NUM_THREADS;       // 64*128/512 = 16
    const int tid       = ty * BLOCK_DIM_X + tx;
    const int numTiles  = (K + TILE_K - 1) / TILE_K;

    for (int by = 0; by < TILES_PER_BLOCK; ++by) {
        for (int bx = 0; bx < TILES_PER_BLOCK; ++bx) {
            const int tileRowBase = (blockIdx.y * TILES_PER_BLOCK + by) * TILE_M;
            const int tileColBase = (blockIdx.x * TILES_PER_BLOCK + bx) * TILE_N;

            float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {{0.f}};

            for (int t = 0; t < numTiles; ++t) {
                const int tileOffset = t * TILE_K;

                #pragma unroll
                for (int i = 0; i < LOADS_A_PER_THREAD; ++i) {
                    const int idx  = tid + i * NUM_THREADS;
                    const int sRow = idx / TILE_K;
                    const int sCol = idx % TILE_K;
                    const int aRow = tileRowBase + sRow;
                    const int aCol = tileOffset + sCol;
                    tileA[sRow][sCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.f;
                }

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

                    // 8 floats = 2 LDS.128
                    float b[COLS_PER_THREAD];
                    const float4 bv0 = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS_PER_THREAD]);
                    const float4 bv1 = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS_PER_THREAD + 4]);
                    b[0] = bv0.x; b[1] = bv0.y; b[2] = bv0.z; b[3] = bv0.w;
                    b[4] = bv1.x; b[5] = bv1.y; b[6] = bv1.z; b[7] = bv1.w;

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
                if (col + 7 < N) {
                    float4 out0 = { sums[r][0], sums[r][1], sums[r][2], sums[r][3] };
                    float4 out1 = { sums[r][4], sums[r][5], sums[r][6], sums[r][7] };
                    *reinterpret_cast<float4*>(&C[row * N + col])     = out0;
                    *reinterpret_cast<float4*>(&C[row * N + col + 4]) = out1;
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

constexpr int MIN_BLOCKS_FOR_TPB2 = 32;

} // namespace

void tile128_launch(const GemmParams& p) {
    assert(static_cast<long long>(p.M) * p.N < static_cast<long long>(INT_MAX));

    if ((p.N & 7) != 0) { bank_pad_vec_launch(p); return; }

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    const int eff2 = TILE_M * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    const int smem_bytes = (TILE_A_ELEMS + TILE_B_ELEMS) * sizeof(float);
    enable_dynamic_smem(tile128_kernel<1>, smem_bytes);
    enable_dynamic_smem(tile128_kernel<2>, smem_bytes);

    if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y);
        tile128_kernel<2><<<grid, block, smem_bytes>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_M - 1) / TILE_M,
                  (p.M + TILE_M - 1) / TILE_M);
        tile128_kernel<1><<<grid, block, smem_bytes>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}
