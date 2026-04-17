#pragma once

// Device-side primitives shared across kernel layers.

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Warp-wide reductions via shuffle
// ---------------------------------------------------------------------------
// Butterfly XOR pattern: after log2(32) = 5 rounds, every lane in the warp
// holds the reduced value (not just lane 0). This matches how online
// softmax / attention consume the result — every lane uses row_max and
// row_sum to rescale its own elements.
//
// Full-warp mask (0xFFFFFFFF). Caller must guarantee all 32 lanes participate;
// passing a partial mask from a divergent branch is undefined behavior.

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset); // Compare against __shfl_down_sync() --> Results are stored only in lane 0
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
    }
    return v;
}

// ---------------------------------------------------------------------------
// cp.async helpers
// ---------------------------------------------------------------------------
// Asynchronous global->shared copy. Requires sm_80+ (Ampere).
//
// Alignment requirements (Undefined Behaviour on violation — caller must enforce):
//   cp.async.4   : both gmem src and smem dst 4-byte aligned
//   cp.async.8   : both 8-byte aligned
//   cp.async.16  : both 16-byte aligned
//
// Cache policy:
//   4B, 8B  -> .ca  (L1 + L2 cached; small transactions benefit from L1)
//   16B     -> .cg  (L2 only; avoids polluting L1 with large tile loads)
//
// Typical usage:
//   cp_async_16B(&smem_tile[i], &gmem[src_off]);   // issue
//   ...more issues...
//   cp_async_commit();                             // group boundary
//   cp_async_wait_group<1>();                      // keep 1 group in flight

#include <cstdint>

__device__ __forceinline__ void cp_async_4B(void* smem_dst, const void* gmem_src) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_8B(void* smem_dst, const void* gmem_src) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                 :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// N = max groups that may still be in flight. wait_group<0> drains everything.
// N must be a compile-time constant (PTX "n" constraint). // Add PTX reference link
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

