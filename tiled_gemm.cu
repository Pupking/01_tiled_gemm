// Layer 0, step 1 beyond naive. Classic SMEM-tiled GEMM:
//   - One thread = one C element (no register tiling yet).
//   - Block computes a BM x BN tile of C, marching over K in BK-wide steps.
//   - Each K-step cooperatively loads A[BM,BK] and B[BK,BN] into SMEM.
// This kernel exists to isolate the SMEM-tiling effect before stacking
// register tiling (multi_tile) and vectorized loads on top.

#include "gemm_common.h"
#include "common/bench_harness.h"

#include <cuda_runtime.h>

namespace {

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 16;

__global__ __launch_bounds__(BM * BN)
void tiled_gemm_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * BM + ty;   // C row this thread owns
    const int col = blockIdx.x * BN + tx;   // C col this thread owns

    float acc = 0.f;

    // March over K in BK-wide tiles. Tail K is handled by bounds-guarding
    // the SMEM load (write 0 for out-of-range); the FMA loop then sees a
    // clean BK-wide tile with zero padding.
    const int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * BK + tx;
        const int b_row = t * BK + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

} // namespace

void tiled_gemm_launch(const GemmParams& p) {
    const dim3 block(BN, BM);
    const dim3 grid((p.N + BN - 1) / BN, (p.M + BM - 1) / BM);
    tiled_gemm_kernel<<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    CUDA_CHECK_LAST();
}

