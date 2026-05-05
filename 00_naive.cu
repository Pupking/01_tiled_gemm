// Baseline: one thread per output element. No SMEM, no tiling.
// Reference point against which every optimization's speedup is measured.
//
// Access pattern: threadIdx.x -> column. Threads in a warp hit consecutive
// B[k, col] elements (coalesced in B) but the same A[row, k] across lanes
// by row-stride (uncoalesced in A). Tiled + coalesced variants fix this.

#include "gemm_common.h"

__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

void naive_launch(const GemmParams& p) {
    constexpr int BS = 16;                                      // 16x16 = 256 threads
    dim3 block(BS, BS);
    dim3 grid((p.N + BS - 1) / BS, (p.M + BS - 1) / BS);
    naive_gemm_kernel<<<grid, block>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    CUDA_CHECK_LAST();
}

