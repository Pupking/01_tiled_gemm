# 01_tiled_gemm - profile-based FP32 GEMM optimization on Ampere

A demonstration of profile-driven kernel optimisation. Six FP32 GEMM kernels
for sm_86 (RTX 3050 Laptop), each step justified by a specific Nsight
Compute counter that moved - **including the one where the predicted
optimisation didn't behave as expected**. Reproducible end-to-end from
the `.ncu-rep` files in this repo.

Full per-step profile summary lives in:
**[docs/01_tiled_gemm.md](../../docs/01_tiled_gemm.md)** , this README is the index.

## Deep Dive Contents

- Reading Speed-of-Light summaries to localise a bottleneck across six kernels.
- Picking specific metrics to *confirm* the issue or optimization that actually helped.
- Reasoning across the L1 / L2 / DRAM hierarchy and the register / SMEM / occupancy optimizations.
- A case where speedup was not due to the expected optimization(useful when doing mulitple optimizations at once).

## Results

|                  | naive | tiled | coalesced | regblock | warp_rebal | bank_pad_vec | cuBLAS |
|------------------|-------|-------|-----------|----------|------------|--------------|--------|
| **ms @ 2048³**   | 40.12 | 11.30 | 9.12      | 7.19     | 6.35       | **5.92**     | 4.31   |
| **GFLOPS**       | 428   | 1520  | 1884      | 2388     | 2705       | **2901**     | 3983   |
| **% cuBLAS**     | 11    | 38    | 47        | 60       | 68         | **73**       | 100    |

| step                                              | what changed                                          | counter that moved          | step gain |
|---------------------------------------------------|-------------------------------------------------------|-----------------------------|-----------|
| [01_tiled.cu](01_tiled.cu)                        | 32×32 SMEM tile, 2×2 register block                   | DRAM throughput 21 -> 43 %   | 3.55 ×    |
| [02_tiled_coalesced.cu](02_tiled_coalesced.cu)    | column-aligned thread mapping for LDG/STG             | st bytes/sector 16 -> 32     | 1.24 ×    |
| [03_regblock.cu](03_regblock.cu)                  | 64×64 tile, 64 outputs/thread (16×4 reg block)        | LSU pipe 95 -> 55 %          | 1.27 ×    |
| [04_warp_rebalance.cu](04_warp_rebalance.cu)      | 128 threads/block, 3 resident blocks/SM               | issue % 53 -> 68             | 1.13 ×    |
| [05_bank_pad_vec.cu](05_bank_pad_vec.cu)          | LDS.128 on tileB inner load (+ pad+1)                 | LSU pipe 89 -> 64 %          | 1.07 ×    |

cuBLAS reference goes through `cublasGemmEx` with `CUBLAS_COMPUTE_32F` +
`CUBLAS_GEMM_DEFAULT` - Tensor Cores disabled, single FP32 accumulator
throughout, for apples-to-apples comparision on CUDA cores. Times are medians of 5 × 50 iterations.

## Experimental Setup

<details>
<summary>Click for more details <code>cudaGetDeviceProperties</code> / <code>cudaDeviceGetAttribute</code> </summary>

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (GA107), sm_86, 16 SMs
- Per-SM: 65,536 registers, 1,536 threads, 100 KB shared memory (48 KB static, 99 KB opt-in), 128 KB unified L1/TEX
- On-chip / off-chip: 1.5 MB L2, 3.68 GB VRAM, 128-bit bus, 192 GB/s peak DRAM
- Compute peak: 128 FP32 lanes × 16 SMs × 2 flops (FMA) × 1.5 GHz = 6.14 TFLOPS FP32
- Toolkit / driver: CUDA 13.0.88, driver 580.82.09, compiled `-O3 --gpu-architecture=sm_86`
- cuBLAS: 13.1.0.3 (ships with CUDA 13.0)
- Shape: M = N = K = 2048. Arithmetic intensity ≈ 341 flops/byte; at 192 GB/s the memory ceiling sits at ≈ 65 TFLOPS, an order of magnitude above compute peak. Compute-bound throughout - DRAM bandwidth never gates the progression.

</details>

## Summary

**Rows 0-3 - break the LSU ceiling.** LSU Pipe utilisation ≥ 94 % across
rows 0-2.

- Row 1 (`tiled`): tile-level reuse - each value in a block's SMEM tile is read by many threads.
- Row 2 (`tiled_coalesced`): coalesced global access. Load count drops further, but LSU stays saturated because LDS + STS traffic per tile-K step still fills most cycles.
- Row 3 (`regblock`, 16×4 outputs / thread): register-level reuse. Each A register feeds 4 output columns, each B register feeds 16 output rows; ≈ 20 SMEM reads supply ≈ 64 FFMAs per inner iteration. LSU collapses 94.7 % -> 55.2 %.
- Row 3's drop is not isolated - register reuse is only wide enough to dominate tile-setup load cost because rows 1-2 optimized the memory access using the tile in SMEM.

**Rows 3-5 - fill the cycles regblock opened.** FMA pipe util climbs
39.7 -> 48.6 -> 56.3 % as the scheduler and SMEM access pattern get fixed.

- Row 4 (`warp_rebalance`): warp-level parallelism. Block doubles 64 -> 128 threads; per-thread register utilization reduces from 255 -> 168 so 3 blocks still fit per SM at the bigger block (3 × 128 × 168 = 64,512 ≤ 65,536 regs/SM, and 3 × 32,768 = 98,304 ≤ 100 KB SMEM - register and SMEM gates bind simultaneously). Issue % rises 53.5 -> 68.4 because each scheduler has more resident warps to pick from when one stalls.
- Row 5 (`bank_pad_vec`): SMEM access-pattern efficiency. LDS.128 on tileB packs 4 SMEM reads into one instruction; the improvement shows up here only because regblock resolved the LSU bottleneck - the same instruction on row 1 or 2 would have been absorbed by still-dominant LSU cost.

**Row 5 -> cuBLAS.** cuBLAS runs ≈ 1.37 × faster than row 5 with L2 hit
82 % vs our 52 %. The gap sits in L2-cache blocking across blocks, not
any per-block technique exercised here.

## Verification

- Cross-checked element-wise against cuBLAS on every launch via the
  `--cross-check` flag (NaN-aware `atol + rtol·|ref|` comparison
  on the host after `cudaMemcpy`-back).
- The cuBLAS reference itself is cross-validated once at M = N = K = 128
  against an FP64 Kahan-summed CPU reference.
- Output buffer is poisoned before every launch so a half-written kernel
  cannot pass verify by reading stale data from a prior cuBLAS call.

## Reproducing

**Build:**
```bash
rm -rf build && mkdir build && cd build
cmake .. && cmake --build . --parallel
cd ..
```

**Run the full Layer-0 sweep (harness timings):**
```bash
./build/bin/gemm_bench --cross-check --M 2048 --N 2048 --K 2048 \
                      --iters 50 --runs 5 --warmup 3
```

**Capture profiles** (Nsight Compute 2025.3+):
```bash
./scripts/profile_layer0.sh
```
Produces one `.ncu-rep` per kernel under `profiles/01_tiled_gemm/`,
plus anchor-metric CSVs under `profiles/01_tiled_gemm/csv/`.

## Scope

- **Per-block, single-shape FP32.** Every number is at M = N = K = 2048.
  Tall, skinny, or rectangular shapes hit different bottlenecks first;
  CUTLASS or cuBLASLt is the right tool there. These codes works for all 
  sizes, but are not optimized for those
- **No Tensor Cores, no mixed precision.** This was mainly to fill the 
issues related to memory and use tiling, with tensor cores, a gap in these levels
will be amplified due to the high compute tensor core offer.
- **Stops at the per-block layer.** The 1.37 × gap to cuBLAS is mainly due to
  cross-block / L2-reuse; the deep-dive doc identifies it using
  (L2 hit 52 % vs cuBLAS 82 %).
