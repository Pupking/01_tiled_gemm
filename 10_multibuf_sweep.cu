// cp.async multi-buffer pipeline, sweeping BUFFERS in {2, 3, 4}.
//
// Same kernel as 11_multibuf, switched to dynamic SMEM via
// cudaFuncSetAttribute so we can use the 99 KB sm_86 carveout instead of
// the 48 KB per-block static limit. That unblocks BUFFERS=3 and 4.
//
// SMEM/block at TILE_K=32: 16,640 B/buffer x BUFFERS.
//   BUFFERS=2: 33 KB -> 3 blocks/SM
//   BUFFERS=3: 50 KB -> 2 blocks/SM
//   BUFFERS=4: 67 KB -> 1 block/SM
// Deeper pipeline trades against strictly fewer resident blocks; the sweep
// pins down where that trade lands on this SKU.

#include "launchers.h"

#include <cassert>
#include <climits>


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

constexpr int TILE_A_ELEMS = TILE_M * TILE_STRIDE_A;
constexpr int TILE_B_ELEMS = TILE_K * TILE_STRIDE_B;

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

template <int BUFFERS, int TILES_PER_BLOCK>
__global__ __launch_bounds__(BLOCK_DIM_X * BLOCK_DIM_Y, 1)
void multibuf_sweep_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    extern __shared__ float smem[];
    float* const tileA_base = smem;
    float* const tileB_base = smem + BUFFERS * TILE_A_ELEMS;

    auto tileA = [&](int b, int r, int c) -> float& {
        return tileA_base[b * TILE_A_ELEMS + r * TILE_STRIDE_A + c];
    };
    auto tileB = [&](int b, int r, int c) -> float& {
        return tileB_base[b * TILE_B_ELEMS + r * TILE_STRIDE_B + c];
    };

    constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;
    constexpr int LOADS_A     = (TILE_M * TILE_K) / NUM_THREADS;
    constexpr int LOADS_B     = (TILE_K * TILE_N) / NUM_THREADS;
    const int tid       = ty * BLOCK_DIM_X + tx;
    const int numTiles  = (K + TILE_K - 1) / TILE_K;

    auto load_tile_async = [&](int kBase, int rowBase, int colBase, int b) {
        #pragma unroll
        for (int i = 0; i < LOADS_A; ++i) {
            const int idx  = tid + i * NUM_THREADS;
            const int sRow = idx / TILE_K;
            const int sCol = idx % TILE_K;
            const int aRow = rowBase + sRow;
            const int aCol = kBase + sCol;
            if (aRow < M && aCol < K) {
                cp_async_4(&tileA(b, sRow, sCol), &A[aRow * K + aCol]);
            } else {
                tileA(b, sRow, sCol) = 0.f;
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
                cp_async_4(&tileB(b, sRow, sCol), &B[bRow * N + bCol]);
            } else {
                tileB(b, sRow, sCol) = 0.f;
            }
        }
    };

    for (int by = 0; by < TILES_PER_BLOCK; ++by) {
        for (int bx = 0; bx < TILES_PER_BLOCK; ++bx) {
            const int tileRowBase = (blockIdx.y * TILES_PER_BLOCK + by) * TILE_M;
            const int tileColBase = (blockIdx.x * TILES_PER_BLOCK + bx) * TILE_N;

            float sums[ROWS_PER_THREAD][COLS_PER_THREAD] = {{0.f}};

            auto compute_tile = [&](int compute_b) {
                #pragma unroll
                for (int k = 0; k < TILE_K; ++k) {
                    float a[ROWS_PER_THREAD];
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r)
                        a[r] = tileA(compute_b, ty * ROWS_PER_THREAD + r, k);

                    const float4 bv = *reinterpret_cast<const float4*>(
                        &tileB(compute_b, k, tx * COLS_PER_THREAD));
                    float b[COLS_PER_THREAD] = { bv.x, bv.y, bv.z, bv.w };

                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                        #pragma unroll
                        for (int c = 0; c < COLS_PER_THREAD; ++c)
                            sums[r][c] += a[r] * b[c];
                    }
                }
            };

            // Pre-fill BUFFERS-1 stages
            #pragma unroll
            for (int b = 0; b < BUFFERS - 1; ++b) {
                if (b < numTiles) {
                    load_tile_async(b * TILE_K, tileRowBase, tileColBase, b);
                }
                cp_async_commit();
            }

            const int main_end = numTiles - (BUFFERS - 1);
            for (int t = 0; t < main_end; ++t) {
                const int load_t = t + BUFFERS - 1;
                load_tile_async(load_t * TILE_K, tileRowBase, tileColBase,
                                load_t % BUFFERS);
                cp_async_commit();

                cp_async_wait_group<BUFFERS - 1>();
                __syncthreads();

                compute_tile(t % BUFFERS);
                __syncthreads();   // race fence before next iter's cp.async write
            }

            // Drain
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

template <int BUFFERS, int TPB>
void launch_one(const GemmParams& p, dim3 grid, dim3 block) {
    const int smem_bytes = BUFFERS * (TILE_A_ELEMS + TILE_B_ELEMS) * sizeof(float);
    enable_dynamic_smem(multibuf_sweep_kernel<BUFFERS, TPB>, smem_bytes);
    multibuf_sweep_kernel<BUFFERS, TPB><<<grid, block, smem_bytes>>>(
        p.dA, p.dB, p.dC, p.M, p.N, p.K);
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
        launch_one<BUFFERS, 4>(p, dim3(grid4x, grid4y), block);
    } else if (grid2x * grid2y >= MIN_BLOCKS_FOR_TPB2) {
        launch_one<BUFFERS, 2>(p, dim3(grid2x, grid2y), block);
    } else {
        launch_one<BUFFERS, 1>(p,
            dim3((p.N + TILE_M - 1) / TILE_M, (p.M + TILE_M - 1) / TILE_M),
            block);
    }
    CUDA_CHECK_LAST();
}

} // namespace

void multibuf_b2_launch(const GemmParams& p) { multibuf_launch_t<2>(p); }
void multibuf_b3_launch(const GemmParams& p) { multibuf_launch_t<3>(p); }
void multibuf_b4_launch(const GemmParams& p) { multibuf_launch_t<4>(p); }
