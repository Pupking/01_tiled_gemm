// cp.async multi-buffered SMEM pipeline (Ampere sm_80+).
//
// While warps compute on tile t, the cp.async DMA loads tile t+1..t+BUFFERS-1
// into other SMEM slots. Load and compute use disjoint HW (LSU vs FMA), so
// the per-K-iter __syncthreads + LSU stall is hidden.
//
// Triggered by 09_sweep at compute SOL 77 % / L1/TEX SOL 67 % - the slack
// lives in the load/compute overlap.
//
// TILE_K=32 (vs 64): SMEM cost is BUFFERS * (tileA + tileB). At TILE_K=64,
// BUFFERS=2 = 66 KB/block = 1 block/SM (occupancy crash). At TILE_K=32,
// BUFFERS=2 = 33 KB/block = 3 blocks/SM, same as gmem_vec. BUFFERS=4 at
// TILE_K=32 would push SMEM > 100 KB; only BUFFERS=2 fits in static SMEM.
// (12_multibuf_sweep covers BUFFERS=2/3/4 via the dynamic-SMEM carveout.)

#include "launchers.h"

#include <cassert>
#include <climits>
#include <cuda_pipeline.h>


namespace {

constexpr int TILE_M          = 64;
constexpr int TILE_N          = 64;
constexpr int TILE_K          = 32;
constexpr int TILE_STRIDE_A   = TILE_K + 1;
constexpr int TILE_STRIDE_B   = TILE_N;
constexpr int BLOCK_DIM_X     = 16;
constexpr int BLOCK_DIM_Y     = 8;
constexpr int ROWS_PER_THREAD = 8;
constexpr int COLS_PER_THREAD = 4;
constexpr int MIN_BLOCKS_FOR_TPB4 = 64;
constexpr int MIN_BLOCKS_FOR_TPB2 = 128;

__device__ __forceinline__ void cp_async_4(float* smem, const float* gmem) {
    uint32_t smem_int = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 : : "r"(smem_int), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

// Issue async loads for one (tileA, tileB) pair into SMEM buffers indexed by `b`.
template <int BUF>
__device__ __forceinline__ void load_tile_async(
    int tid, int kBase, int rowBase, int colBase,
    int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B,
    float (&tileA)[BUF][TILE_M][TILE_STRIDE_A],
    float (&tileB)[BUF][TILE_K][TILE_STRIDE_B],
    int b) {

    constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;
    constexpr int LOADS_A = (TILE_M * TILE_K) / NUM_THREADS;  // 64*32/128 = 16
    constexpr int LOADS_B = (TILE_K * TILE_N) / NUM_THREADS;  // 32*64/128 = 16

    #pragma unroll
    for (int i = 0; i < LOADS_A; ++i) {
        const int idx  = tid + i * NUM_THREADS;
        const int sRow = idx / TILE_K;
        const int sCol = idx % TILE_K;
        const int aRow = rowBase + sRow;
        const int aCol = kBase + sCol;
        if (aRow < M && aCol < K) {
            cp_async_4(&tileA[b][sRow][sCol], &A[aRow * K + aCol]);
        } else {
            tileA[b][sRow][sCol] = 0.f;
        }
    }

    #pragma unroll
    for (int i = 0; i < LOADS_B; ++i) {
        const int idx  = tid + i * NUM_THREADS;
        const int sRow = idx / TILE_N;
        const int sCol = idx % TILE_N;
        const int bRow = kBase + sRow;
        const int bCol = colBase + sCol;
        if (bRow < K && bCol < N) {
            cp_async_4(&tileB[b][sRow][sCol], &B[bRow * N + bCol]);
        } else {
            tileB[b][sRow][sCol] = 0.f;
        }
    }
}

template <int BUFFERS, int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 3)
void multibuf_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float tileA[BUFFERS][TILE_M][TILE_STRIDE_A];
    __shared__ float tileB[BUFFERS][TILE_K][TILE_STRIDE_B];

    const int tid = ty * BLOCK_DIM_X + tx;
    const int numTiles = (K + TILE_K - 1) / TILE_K;

    for (int by = 0; by < TILES_PER_BLOCK; ++by) {
        for (int bx = 0; bx < TILES_PER_BLOCK; ++bx) {
            const int tileRowBase = (blockIdx.y * TILES_PER_BLOCK + by) * TILE_M;
            const int tileColBase = (blockIdx.x * TILES_PER_BLOCK + bx) * TILE_N;

            float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {{0.f}};

            // Inline lambda for the per-tile compute on a given buffer slot.
            auto compute_tile = [&](int compute_b) {
                #pragma unroll
                for (int k = 0; k < TILE_K; ++k) {
                    float a[ROWS_PER_THREAD];
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r)
                        a[r] = tileA[compute_b][ty * ROWS_PER_THREAD + r][k];

                    const float4 bv = *reinterpret_cast<const float4*>(&tileB[compute_b][k][tx * COLS_PER_THREAD]);
                    float b[COLS_PER_THREAD] = { bv.x, bv.y, bv.z, bv.w };

                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                        #pragma unroll
                        for (int c = 0; c < COLS_PER_THREAD; ++c)
                            sums[r][c] += a[r] * b[c];
                    }
                }
            };

            // Pre-fill BUFFERS-1 stages of the pipeline.
            #pragma unroll
            for (int b = 0; b < BUFFERS - 1; ++b) {
                if (b < numTiles) {
                    load_tile_async(tid, b * TILE_K, tileRowBase, tileColBase,
                                    M, N, K, A, B, tileA, tileB, b);
                }
                cp_async_commit();
            }

            // Steady-state pipeline: at iter t, issue the load that will be
            // needed BUFFERS-1 iters later, then wait for tile t's load to
            // complete (issued BUFFERS-1 iters ago). Stop the loop as soon
            // as there are no more loads to issue.
            const int main_end = numTiles - (BUFFERS - 1);
            for (int t = 0; t < main_end; ++t) {
                const int load_t = t + BUFFERS - 1;
                const int load_b = load_t % BUFFERS;
                load_tile_async(tid, load_t * TILE_K, tileRowBase, tileColBase,
                                M, N, K, A, B, tileA, tileB, load_b);
                cp_async_commit();

                cp_async_wait_group<BUFFERS - 1>();
                __syncthreads();

                compute_tile(t % BUFFERS);

                // Sync after compute so the *next* iter's load doesn't start
                // writing to the buffer the slow warps may still be reading
                // from. Without this, fast warps issue cp.async into the
                // buffer slow warps haven't finished consuming -> race.
                __syncthreads();
            }

            // Drain phase: wait for any remaining pending loads and compute
            // the last BUFFERS-1 tiles. No more loads, so wait_group<0>.
            cp_async_wait_group<0>();
            __syncthreads();
            const int drain_start = (main_end < 0) ? 0 : main_end;
            for (int t = drain_start; t < numTiles; ++t) {
                compute_tile(t % BUFFERS);
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

template <int BUFFERS>
void multibuf_launch_t(const GemmParams& p) {
    assert(static_cast<long long>(p.M) * p.N < static_cast<long long>(INT_MAX));

    if ((p.N & 3) != 0) { bank_pad_vec_launch(p); return; }

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);

    const int eff4   = TILE_M * 4;
    const int grid4x = (p.N + eff4 - 1) / eff4;
    const int grid4y = (p.M + eff4 - 1) / eff4;

    const int eff2   = TILE_M * 2;
    const int grid2x = (p.N + eff2 - 1) / eff2;
    const int grid2y = (p.M + eff2 - 1) / eff2;

    if (grid4x * grid4y >= MIN_BLOCKS_FOR_TPB4) {
        dim3 grid(grid4x, grid4y);
        multibuf_kernel<BUFFERS, 4><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        dim3 grid(grid2x, grid2y);
        multibuf_kernel<BUFFERS, 2><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        dim3 grid((p.N + TILE_M - 1) / TILE_M,
                  (p.M + TILE_M - 1) / TILE_M);
        multibuf_kernel<BUFFERS, 1><<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}

} // namespace

void multibuf2_launch(const GemmParams& p) { multibuf_launch_t<2>(p); }
// BUFFERS=3 would need 49,920 B/block static SMEM, over the 48 KB per-block
// static limit. The dynamic-SMEM variant in 10_multibuf_sweep.cu provides
// it as `multibuf_b3` - see that file.
