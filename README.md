# 01_tiled_gemm - profile-driven FP32 GEMM optimization on Ampere

Six FP32 GEMM kernels for sm_86 (RTX 3050 Laptop) walked from a one-line
naive baseline to ~74 % of cuBLAS, each step justified by the specific
Nsight Compute counter that moved -- including one step where the
counter that improved was not the one we predicted. Reproducible
end-to-end from the `.ncu-rep` files in this repo.

The per-step profile deep-dive is at
[docs/01_tiled_gemm.md](docs/01_tiled_gemm.md); this README is the index.

The repo also extends past those six baselines into a wider exploration
(supertile remap, vectorised LDG/STG, multi-buffered cp.async, split-K,
trait-templated kernel family) -- run `gemm_bench` to see all of them.

## Results

Representative run on the setup below; regenerate with the command in
[Reproducing](#reproducing).

|                  | naive | tiled | coalesced | regblock | warp_rebal | bank_pad_vec | cuBLAS |
|------------------|-------|-------|-----------|----------|------------|--------------|--------|
| **ms @ 2048^3**  | 40.78 | 11.59 | 9.33      | 7.40     | 6.53       | **6.01**     | 4.45   |
| **GFLOPS**       | 421   | 1483  | 1842      | 2322     | 2633       | **2857**     | 3861   |
| **% cuBLAS**     | 11    | 38    | 48        | 60       | 68         | **74**       | 100    |

| step                                              | what changed                                       | counter that moved         | step gain |
|---------------------------------------------------|----------------------------------------------------|----------------------------|-----------|
| [01_tiled.cu](01_tiled.cu)                        | 32x32 SMEM tile, 2x2 register block                | DRAM throughput 21 -> 43 % | 3.52x     |
| [02_tiled_coalesced.cu](02_tiled_coalesced.cu)    | column-aligned thread mapping for LDG/STG          | st bytes/sector 16 -> 32   | 1.24x     |
| [03_regblock.cu](03_regblock.cu)                  | 64x64 tile, 64 outputs/thread (16x4 reg block)     | LSU pipe 95 -> 55 %        | 1.26x     |
| [04_warp_rebalance.cu](04_warp_rebalance.cu)      | 128 threads/block, 3 resident blocks/SM            | issue % 53 -> 68           | 1.13x     |
| [05_bank_pad_vec.cu](05_bank_pad_vec.cu)          | LDS.128 on tileB inner load (+ pad+1)              | LSU pipe 89 -> 64 %        | 1.09x     |

cuBLAS reference goes through `cublasGemmEx` with `CUDA_R_32F`
operands/results, `CUBLAS_COMPUTE_32F`, and `CUBLAS_GEMM_DEFAULT`.
The handwritten kernels are FP32 CUDA-core kernels; the tracked cuBLAS
profile for this run reports 0 % Tensor pipe utilization. Times are
medians of 5 x 50 iterations; the command to regenerate them is below.

## Experimental Setup

<details>
<summary>Click for more details <code>cudaGetDeviceProperties</code> / <code>cudaDeviceGetAttribute</code> </summary>

- GPU: NVIDIA GeForce RTX 3050 Laptop (GA107), sm_86, 16 SMs
- Per-SM: 65,536 registers, 1,536 threads, 100 KB shared memory (48 KB static, 99 KB opt-in), 128 KB unified L1/TEX
- On-chip / off-chip: 1.5 MB L2, 3.68 GB VRAM, 128-bit bus, 192 GB/s peak DRAM
- Compute peak: 128 FP32 lanes * 16 SMs * 2 flops (FMA) * 1.5 GHz = 6.14 TFLOPS FP32
- Toolkit / driver: CUDA 13.0.88, driver 580.82.09, compiled `-O3 --gpu-architecture=sm_86`
- cuBLAS: 13.1.0.3 (ships with CUDA 13.0)
- Shape: M = N = K = 2048. Arithmetic intensity ~341 flops/byte; at 192 GB/s the memory ceiling sits at ~65 TFLOPS, an order of magnitude above compute peak. Compute-bound throughout - DRAM bandwidth never gates the progression.

</details>

## Summary

**Rows 0-3 -- break the LSU ceiling.** LSU Pipe utilisation >= 94 % across rows 0-2.

- Row 1 (`tiled`): tile-level reuse - each value in a block's SMEM tile is read by many threads.
- Row 2 (`tiled_coalesced`): coalesced global access. Load count drops further, but LSU stays saturated because LDS + STS traffic per tile-K step still fills most cycles.
- Row 3 (`regblock`, 16x4 outputs / thread): register-level reuse. Each A register feeds 4 output columns, each B register feeds 16 output rows; ~20 SMEM reads supply ~64 FFMAs per inner iteration. LSU collapses 94.7 % -> 55.2 %.
- Row 3's drop is not isolated -- register reuse is only wide enough to dominate tile-setup load cost because rows 1-2 fixed the SMEM access pattern.

**Rows 3-5 -- fill the cycles regblock opened.** FMA pipe util climbs 39.7 -> 48.6 -> 56.3 % as the scheduler and SMEM access pattern get fixed.

- Row 4 (`warp_rebalance`): warp-level parallelism. Block doubles 64 -> 128 threads; per-thread regs drop 255 -> 168 so 3 blocks still fit per SM at the bigger block (3 * 128 * 168 = 64,512 <= 65,536 regs/SM; 3 * 32,768 = 98,304 <= 100 KB SMEM - register and SMEM gates bind simultaneously). Issue % rises 53.5 -> 68.4 because each scheduler has more resident warps to pick from when one stalls.
- Row 5 (`bank_pad_vec`): SMEM access-pattern efficiency. LDS.128 on tileB packs 4 SMEM reads into one instruction; the improvement shows up here only because regblock resolved the LSU bottleneck - the same instruction at row 1 or 2 would have been absorbed by still-dominant LSU cost.

**Row 5 -> cuBLAS.** cuBLAS runs ~1.37x faster than row 5 with L2 hit 82 % vs our 52 %. The gap sits in L2-cache blocking across blocks, not any per-block technique exercised here.

## Verification

- Element-wise against cuBLAS on every launch (NaN-aware `atol + rtol*|ref|` comparison on the host after `cudaMemcpy` back).
- cuBLAS itself is cross-validated once at M = N = K = 128 against an FP64 Kahan-summed CPU reference via the `--cross-check` flag.
- Output buffer is poisoned (0xFF -> NaN) before every launch so a half-written kernel cannot pass verify by leaving stale data behind.

## Reproducing

Build:
```bash
cmake -S . -B build && cmake --build build --parallel -j
```

Run the full sweep:
```bash
./build/bin/gemm_bench --cross-check --M 2048 --N 2048 --K 2048 \
                      --iters 50 --runs 5 --warmup 5
```

Capture profiles (Nsight Compute 2025.3+):
```bash
./scripts/profile_layer0.sh
```
Produces one `.ncu-rep` per kernel under `profiles/` for opening in Nsight
Compute or exporting as metric CSVs.

## Scope

- **Per-block, single-shape FP32.** Every number is at M = N = K = 2048. Tall, skinny, or rectangular shapes hit different bottlenecks first; CUTLASS or cuBLASLt is the right tool there. These kernels work for all sizes but are tuned only for this one.
- **FP32 CUDA-core only.** No mixed precision, no WMMA/MMA/GMMA, no Tensor Cores.
- **Stops at the per-block layer.** The 1.37x gap to cuBLAS is mainly cross-block / L2 reuse; the deep-dive identifies it via L2 hit 52 % vs cuBLAS 82 %.
