// 64x128 tile - symmetric mirror of wider_m's 128x64.
//
// Block (32, 8) instead of (16, 16): BX=32 means a warp's threads all share
// the same ty (broadcast on tileA reads) and the inner b-load covers 32
// unique tx values, doubling LDS.128 throughput on tileB vs wider_m.

#include "launchers.h"
#include <cassert>
#include <climits>


namespace {

constexpr int TILE_M = 64, TILE_N = 128, TILE_K = 64;
constexpr int TILE_STRIDE_A = TILE_K + 1;
constexpr int TILE_STRIDE_B = TILE_N;
constexpr int BX = 32, BY = 8;
constexpr int ROWS = 8, COLS = 4;
constexpr int TILE_A_ELEMS = TILE_M * TILE_STRIDE_A;
constexpr int TILE_B_ELEMS = TILE_K * TILE_STRIDE_B;

template <int TPB>
__global__ __launch_bounds__(BX * BY, 2)
void wider_n_kernel(const float* __restrict__ A, const float* __restrict__ B,
                    float* __restrict__ C, int M, int N, int K) {
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * BX + tx;
    extern __shared__ float smem[];
    float (*tileA)[TILE_STRIDE_A] = reinterpret_cast<float (*)[TILE_STRIDE_A]>(smem);
    float (*tileB)[TILE_STRIDE_B] = reinterpret_cast<float (*)[TILE_STRIDE_B]>(smem + TILE_A_ELEMS);

    constexpr int NUM_THREADS = BX * BY;             // 256
    constexpr int LOADS_A = (TILE_M * TILE_K) / NUM_THREADS;  // 16
    constexpr int LOADS_B = (TILE_K * TILE_N) / NUM_THREADS;  // 32
    const int numTiles = (K + TILE_K - 1) / TILE_K;

    for (int by_ = 0; by_ < TPB; ++by_) {
    for (int bx_ = 0; bx_ < TPB; ++bx_) {
        const int rowBase = (blockIdx.y * TPB + by_) * TILE_M;
        const int colBase = (blockIdx.x * TPB + bx_) * TILE_N;
        float sums[ROWS][COLS] = {{0.f}};

        for (int t = 0; t < numTiles; ++t) {
            const int kOff = t * TILE_K;
            #pragma unroll
            for (int i = 0; i < LOADS_A; ++i) {
                const int idx = tid + i * NUM_THREADS;
                const int sR = idx / TILE_K, sC = idx % TILE_K;
                const int aR = rowBase + sR, aC = kOff + sC;
                tileA[sR][sC] = (aR < M && aC < K) ? A[aR * K + aC] : 0.f;
            }
            #pragma unroll
            for (int i = 0; i < LOADS_B; ++i) {
                const int idx = tid + i * NUM_THREADS;
                const int sR = idx / TILE_N, sC = idx % TILE_N;
                const int bR = kOff + sR, bC = colBase + sC;
                tileB[sR][sC] = (bR < K && bC < N) ? B[bR * N + bC] : 0.f;
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                float a[ROWS];
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) a[r] = tileA[ty * ROWS + r][k];
                const float4 bv = *reinterpret_cast<const float4*>(&tileB[k][tx * COLS]);
                float b[COLS] = { bv.x, bv.y, bv.z, bv.w };
                #pragma unroll
                for (int r = 0; r < ROWS; ++r)
                #pragma unroll
                for (int c = 0; c < COLS; ++c) sums[r][c] += a[r] * b[c];
            }
            __syncthreads();
        }

        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rowBase + ty * ROWS + r;
            if (row >= M) continue;
            const int col = colBase + tx * COLS;
            if (col + 3 < N) {
                float4 out = { sums[r][0], sums[r][1], sums[r][2], sums[r][3] };
                *reinterpret_cast<float4*>(&C[row * N + col]) = out;
            } else {
                #pragma unroll
                for (int c = 0; c < COLS; ++c) if (col + c < N) C[row * N + col + c] = sums[r][c];
            }
        }
    }}
}

} // namespace

void wider_n_launch(const GemmParams& p) {
    if ((p.N & 3) != 0) { bank_pad_vec_launch(p); return; }
    dim3 block(BX, BY);
    const int eff2_m = TILE_M * 2, eff2_n = TILE_N * 2;
    const int g2x = (p.N + eff2_n - 1) / eff2_n, g2y = (p.M + eff2_m - 1) / eff2_m;
    const int smem_bytes = (TILE_A_ELEMS + TILE_B_ELEMS) * sizeof(float);
    enable_dynamic_smem(wider_n_kernel<1>, smem_bytes);
    enable_dynamic_smem(wider_n_kernel<2>, smem_bytes);
    if (g2x * g2y >= 64) {
        wider_n_kernel<2><<<dim3(g2x, g2y), block, smem_bytes>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    } else {
        wider_n_kernel<1><<<dim3((p.N + TILE_N - 1) / TILE_N, (p.M + TILE_M - 1) / TILE_M), block, smem_bytes>>>(p.dA, p.dB, p.dC, p.M, p.N, p.K);
    }
    CUDA_CHECK_LAST();
}
