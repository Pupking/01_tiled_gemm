// ldg_vec geometry + cp.async double buffer. cp.async writes gmem -> smem
// directly, replacing each LDG.128 + STS.128 pair with one cp.async.cg.16.

#include "launchers.h"
#include "launch_helpers.h"


namespace {

constexpr int TILE_M = 128, TILE_N = 64, TILE_K = 64;
// cp.async.cg requires 16-byte alignment on the SMEM dest. TILE_K=64 floats
// per row already = 256 B (16-aligned), so no padding needed. Any +4 padding
// would push 2-buffer SMEM to 100 KB, over the SM 8.6 99 KB opt-in cap.
constexpr int TILE_STRIDE_A = TILE_K;
constexpr int TILE_STRIDE_B = TILE_N;
constexpr int BX = 16, BY = 16;
constexpr int ROWS = 8, COLS = 4;
constexpr int TILE_A_ELEMS = TILE_M * TILE_STRIDE_A;
constexpr int TILE_B_ELEMS = TILE_K * TILE_STRIDE_B;
constexpr int BUFFERS = 2;

__device__ __forceinline__ void cp_async_16(float* smem, const float* gmem) {
    uint32_t s = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

template <int TPB>
__global__ __launch_bounds__(BX * BY, 1)
void ldg_async_kernel(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ C, int M, int N, int K) {
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * BX + tx;

    extern __shared__ float smem[];
    float* tA[BUFFERS];
    float* tB[BUFFERS];
    #pragma unroll
    for (int b = 0; b < BUFFERS; ++b) {
        tA[b] = smem + b * (TILE_A_ELEMS + TILE_B_ELEMS);
        tB[b] = tA[b] + TILE_A_ELEMS;
    }
    auto A_at = [&](int b, int r, int c) -> float& { return tA[b][r * TILE_STRIDE_A + c]; };
    auto B_at = [&](int b, int r, int c) -> float& { return tB[b][r * TILE_STRIDE_B + c]; };

    constexpr int NT = BX * BY;
    constexpr int LA4 = (TILE_M * TILE_K / 4) / NT;  // 8
    constexpr int LB4 = (TILE_K * TILE_N / 4) / NT;  // 4
    const int numTiles = K / TILE_K;

    auto load_async = [&](int kOff, int rowBase, int colBase, int b) {
        #pragma unroll
        for (int i = 0; i < LA4; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / (TILE_K / 4);
            const int sC = (idx % (TILE_K / 4)) * 4;
            cp_async_16(&A_at(b, sR, sC), &A[(rowBase + sR) * K + kOff + sC]);
        }
        #pragma unroll
        for (int i = 0; i < LB4; ++i) {
            const int idx = tid + i * NT;
            const int sR = idx / (TILE_N / 4);
            const int sC = (idx % (TILE_N / 4)) * 4;
            cp_async_16(&B_at(b, sR, sC), &B[(kOff + sR) * N + colBase + sC]);
        }
    };

    for (int by_ = 0; by_ < TPB; ++by_) {
    for (int bx_ = 0; bx_ < TPB; ++bx_) {
        const int rowBase = (blockIdx.y * TPB + by_) * TILE_M;
        const int colBase = (blockIdx.x * TPB + bx_) * TILE_N;
        float sums[ROWS][COLS] = {{0.f}};

        // Pre-fill buffer 0
        load_async(0, rowBase, colBase, 0);
        cp_async_commit();

        const int main_end = numTiles - 1;
        for (int t = 0; t < main_end; ++t) {
            load_async((t + 1) * TILE_K, rowBase, colBase, (t + 1) % BUFFERS);
            cp_async_commit();
            cp_async_wait_group<1>();
            __syncthreads();

            const int cb = t % BUFFERS;
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                float a[ROWS];
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) a[r] = A_at(cb, ty * ROWS + r, k);
                const float4 bv = *reinterpret_cast<const float4*>(&B_at(cb, k, tx * COLS));
                float b[COLS] = { bv.x, bv.y, bv.z, bv.w };
                #pragma unroll
                for (int r = 0; r < ROWS; ++r)
                #pragma unroll
                for (int c = 0; c < COLS; ++c) sums[r][c] += a[r] * b[c];
            }
            __syncthreads();
        }
        // Drain
        cp_async_wait_group<0>();
        __syncthreads();
        const int cb = (numTiles - 1) % BUFFERS;
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) a[r] = A_at(cb, ty * ROWS + r, k);
            const float4 bv = *reinterpret_cast<const float4*>(&B_at(cb, k, tx * COLS));
            float b[COLS] = { bv.x, bv.y, bv.z, bv.w };
            #pragma unroll
            for (int r = 0; r < ROWS; ++r)
            #pragma unroll
            for (int c = 0; c < COLS; ++c) sums[r][c] += a[r] * b[c];
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = rowBase + ty * ROWS + r;
            const int col = colBase + tx * COLS;
            float4 out = { sums[r][0], sums[r][1], sums[r][2], sums[r][3] };
            *reinterpret_cast<float4*>(&C[row * N + col]) = out;
        }
    }}
}

} // namespace

void ldg_async_launch(const GemmParams& p) {
    if ((p.M % TILE_M) || (p.N % TILE_N) || (p.K % TILE_K) || (p.N & 3)) {
        bank_pad_vec_launch(p); return;
    }
    const int smem_bytes = BUFFERS * (TILE_A_ELEMS + TILE_B_ELEMS) * sizeof(float);
    launch::grid2d_tpb(p, dim3(BX, BY), smem_bytes,
                       TILE_M, TILE_N,
                       ldg_async_kernel<1>, ldg_async_kernel<2>);
}
