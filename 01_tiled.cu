// Layer 0 — SMEM tiling + 2×2 register sub-tile per thread.
//
// Structure:
//   - 32×32 output tile per block.
//   - 16×16 = 256 threads (8 warps), each owning a 2×2 output sub-tile.
//   - Cooperative tile load: 256 threads × 4 loads = 32×32 = 1024 A/B elems.
//   - Inner k-loop reads 2 A rows and 2 B cols into registers and updates
//     4 accumulators, turning 4 SMEM-fed FMAs into 4 register-fed FMAs per
//     k step — the first place GMEM bandwidth stops being the bottleneck.

#include "gemm_common.h"

namespace {

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_DIM = 16;  // 16x16 = 256 threads

__global__ __launch_bounds__(BLOCK_DIM * BLOCK_DIM)
void tiled_gemm_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row0 = blockIdx.y * TILE_SIZE + ty * 2;
    const int col0 = blockIdx.x * TILE_SIZE + tx * 2;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float s00 = 0.f, s01 = 0.f, s10 = 0.f, s11 = 0.f;

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int tileOffset = t * TILE_SIZE;
        const int tid = ty * BLOCK_DIM + tx;  // 0..255

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx  = tid + i * 256;
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
            const float a0 = tileA[ty * 2    ][k];
            const float a1 = tileA[ty * 2 + 1][k];
            const float b0 = tileB[k][tx * 2    ];
            const float b1 = tileB[k][tx * 2 + 1];
            s00 += a0 * b0;
            s01 += a0 * b1;
            s10 += a1 * b0;
            s11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0     < M && col0     < N) C[(row0)     * N + col0    ] = s00;
    if (row0     < M && col0 + 1 < N) C[(row0)     * N + col0 + 1] = s01;
    if (row0 + 1 < M && col0     < N) C[(row0 + 1) * N + col0    ] = s10;
    if (row0 + 1 < M && col0 + 1 < N) C[(row0 + 1) * N + col0 + 1] = s11;
}

} // namespace

void tiled_launch(const GemmParams& p) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((p.N + TILE_SIZE - 1) / TILE_SIZE,
              (p.M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gemm_kernel<<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    CUDA_CHECK_LAST();
}
