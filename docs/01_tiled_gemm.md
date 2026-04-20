# 01_tiled_gemm - per-kernel profile details

Summary is present in: [README](../README.md).

All profiles should be read in the following order:

First get a glance over the performance to identify the area of issue
```bash
ncu --import < kernel >.ncu-rep --page details --section SpeedOfLight -k "<kernel_name>"
```
Based on the issue look at details in each section or metrics directly, examples of these are highlighted below:
```bash
ncu --import < kernel >.ncu-rep --page details --section < section_of_choice > -k "<kernel_name>"
```
Here section_of_choice can be(this is a limited set, for more read the Nsight Compute docs):
- SpeedOfLight
- ComputeWorkloadAnalysis
- MemoryWorkloadAnalysis
- LaunchStats
- Occupancy
- WarpStateStats
- InstructionStats
- WorkloadDistribution
- SourceCounters

Finally to confirm the inefficiency look at the source metrics.
```bash
ncu --import < kernel >.ncu-rep --page raw --metrics < metrics > -k "<kernel_name>"
```
Here metrics can include multiple metrics, add them using a ','.
There are many metrics, for each type of issue we might have to verify with one or more.
Metrics used will be shown with each kernel.

A simpler way is to just use ncu-ui. The above are only if you do not want to or cannot leave the terminal.


## 0. naive - [00_naive.cu](../00_naive.cu)

One thread per output element, no reuse, no tiling. This code is fairly simple, you can just read the code and add a shared memory optimization and it will be faster. The following shows one reason why it is slow and which hardware unit is stressed.

**Speed of Light Throughput Summary**
naive_gemm_kernel(const float *, const float *, float *, int, int, int) (128, 128, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          5.99
    SM Frequency                    Ghz          1.24
    Elapsed Cycles                cycle    68,126,956
    Memory Throughput                 %         98.57
    DRAM Throughput                   %         20.87
    Duration                         ms         55.05
    L1/TEX Cache Throughput           %         98.60
    L2 Cache Throughput               %         17.57
    SM Active Cycles              cycle 68,108,156.75
    Compute (SM) Throughput           %         98.57
    ----------------------- ----------- -------------

- The memory throughput is 98.57%, but DRAM throughput not so much (it is at 20.87%).
- L1/TEX Cache is being used the most. We need to reduce the work it does that is transfer memory to some other unit.
- This can be resolved using shared memory tiling, to having coalesced loads and stores.
- Just to show the main issue the following shows the metric to check to confirm that LSU(Load Store Unit) pipe is the issue.

**Metrics to confirm LSU pipe bottleneck**

Use these metrics:
- sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
- sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
- smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio

  naive_gemm_kernel(const float *, const float *, float *, int, int, int) (128, 128, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                                                     Metric Unit         Metric Value
  --------------------------------------------------------------- ----------- --------------------
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active               %                98.60
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed              %                19.26
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio byte/sector                   18

- LSU ~ 5 x FMA: load-store is the ceiling.
- The last metric shows the uncoalesced nature, this should be 32.

---

## 1. tiled - [01_tiled.cu](../01_tiled.cu)

32x32 SMEM tile loaded cooperatively per tile-K step; 2x2 register tile
per thread; 256 threads / block. You can still find a few inefficiency from the code, like uncoalesced stores. Lets look at this more closely.

**Speed of Light Throughput Summary**

<unnamed>::tiled_gemm_kernel(const float *, const float *, float *, int, int, int) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          5.99
    SM Frequency                    Ghz          1.24
    Elapsed Cycles                cycle    17,482,887
    Memory Throughput                 %         96.15
    DRAM Throughput                   %         42.78
    Duration                         ms         14.13
    L1/TEX Cache Throughput           %         96.19
    L2 Cache Throughput               %         33.71
    SM Active Cycles              cycle 17,475,989.94
    Compute (SM) Throughput           %         96.15
    ----------------------- ----------- -------------

- Compute-SM, Memory, and L1/TEX all at 96% throughput - still near the ceiling naive hit.
- The shift is downstream in the hierarchy: DRAM throughput doubles vs naive (20.9 -> 42.8%) and L2 throughput doubles (17.6 -> 33.7%). Memory pipe is now actively moving data through the hierarchy instead of absorbing it in L1.
- One main metric apart from runtime to look at is the number of Elapsed cycles, the reduction is around 3.8x.
- This reduction is due to the decrease in the number of sectors requested vs accessed(in metrics).
- Before that lets look at the Memory workload analysis.

**Memory Workload Analysis**

<unnamed>::tiled_gemm_kernel(const float *, const float *, float *, int, int, int) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Memory Workload Analysis
    -------------------------------------- ----------- ------------
    Metric Name                            Metric Unit Metric Value
    -------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                0
    Local Memory Spilling Request Overhead           %            0
    Memory Throughput                          Gbyte/s        81.97
    Mem Busy                                         %        84.69
    Max Bandwidth                                    %        96.13
    L1/TEX Hit Rate                                  %         0.70
    L2 Persisting Size                           Kbyte       294.91
    L2 Compression Success Rate                      %            0
    L2 Compression Ratio                                          0
    L2 Compression Input Sectors                sector            0
    L2 Hit Rate                                      %        47.34
    Mem Pipes Busy                                   %        96.13
    -------------------------------------- ----------- ------------

- L1 hit = 0.7%: each global sector is loaded once into SMEM and then read only from SMEM - L1 is effectively bypassed.
- L2 hit = 47% (slightly below naive: fewer redundant global reads means less cross-block L2 overlap).
- Mem Pipes busy is still high, still a memory related issue.

**Metrics to confirm the issue**

Use these metrics:
- sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
- sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
- smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio
- smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio

<unnamed>::tiled_gemm_kernel(const float *, const float *, float *, int, int, int) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                                                     Metric Unit         Metric Value
  --------------------------------------------------------------- ----------- --------------------
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active               %                96.20
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed              %                28.50
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio byte/sector                   32
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio byte/sector                   16


- LSU ~ 3.5 x FMA: load-store is still an issue.
- LSU barely moves: global loads have been replaced by SMEM loads at ~1 : 1 in the inner loop.
- The last metric shows the uncoalesced nature, this should be 32.
- FMA gains come from the absolute load-count drop and from LDG latency not being such a big problem - as this changes to shared memory latency which is considerably faster.
- L1 hit collapses because reuse relocates to SMEM.
- Loads are now coalesced.
- Stores are not, stride is 2, hence only 16 bytes are used per sector instead of 32.

## 2. tiled_coalesced - [02_tiled_coalesced.cu](../02_tiled_coalesced.cu)

32x32 SMEM tile; register tile grows to 8x2 (16 outputs / thread);
64 threads / block. Coalesced global access - each warp's 32 lanes hit
a contiguous sector per LDG, STG

**Speed of Light Throughput Summary**
<unnamed>::tiled_coalesced_gemm_kernel(const float *, const float *, float *, int, int, int) (64, 64, 1)x(16, 4, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          5.99
    SM Frequency                    Ghz          1.24
    Elapsed Cycles                cycle    13,383,069
    Memory Throughput                 %         94.68
    DRAM Throughput                   %         54.18
    Duration                         ms         10.81
    L1/TEX Cache Throughput           %         94.73
    L2 Cache Throughput               %         38.91
    SM Active Cycles              cycle 13,373,910.75
    Compute (SM) Throughput           %         94.68
    ----------------------- ----------- -------------

- Memory and compute SOL both at 94.7%: same ceiling as naive / tiled.
- DRAM throughput 54.2% (vs tiled's 42.8%) and L2 throughput 38.9% (vs 33.7%): coalesced LDG/STG moves more bytes per cycle through the memory hierarchy.
- For this particular code, the profile does not give sufficient proof for any particular issue, by getting per section data.
- We need to look at the metrics. Also if you look at the code, you may see that the work done per tile/warp can be increased(increase register reuse).

**Core Metrics and per warp work**

Metrics used are:
- sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
- sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
- smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio byte/sector
- smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio byte/sector
- derived__avg_thread_executed
- launch__registers_per_thread

The first set of metrics are just to confirm that we have coalesced loads and stores. Along with the FMA and LSU pipe utilization.

<unnamed>::tiled_coalesced_gemm_kernel(const float *, const float *, float *, int, int, int) (64, 64, 1)x(16, 4, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                                                     Metric Unit         Metric Value
  --------------------------------------------------------------- ----------- --------------------
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active               %                94.75
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed              %                35.55
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio byte/sector                   32
  smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio byte/sector                   32

- LSU ~ 2.6 x FMA.
- Lets look at the per thread work: ROWS_PER_THREAD × COLS_PER_THREAD = 8 × 2 = 16 // From code, so each works on 512 elements.
- The per thread work could be increased, this is what is done on the next optimization. Also we can expect the number of registers used to increase.

<unnamed>::tiled_coalesced_gemm_kernel(const float *, const float *, float *, int, int, int) (64, 64, 1)x(16, 4, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                      Metric Unit         Metric Value
  ---------------------------- --------------- --------------------
  derived__avg_thread_executed          thread               39,616
  launch__registers_per_thread register/thread                  168

These metrics can be compared with the next code to get the idea that work per thread is increased and so is register reuse.
But these are not always the right values to check, they might just indicate bad code.

## 3. regblock - [03_regblock.cu](../03_regblock.cu)

64x64 SMEM tile; register tile grows to 16x4 (64 outputs / thread);
16x4 = 64 threads / block. Adaptive `TILES_PER_BLOCK = 2` at 2048^3 so
each block produces a 128x128 output region.

**Speed of Light Throughput Summary**
void <unnamed>::multi_tile_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 4, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          5.99
    SM Frequency                    Ghz          1.24
    Elapsed Cycles                cycle    11,562,765
    Memory Throughput                 %         54.58
    DRAM Throughput                   %         29.81
    Duration                         ms          9.34
    L1/TEX Cache Throughput           %         55.16
    L2 Cache Throughput               %         25.61
    SM Active Cycles              cycle 11,436,864.62
    Compute (SM) Throughput           %         54.58
    ----------------------- ----------- -------------

- Memory and compute SOL both drop to ~55% (from ~95% previously).
- DRAM throughput halves (54.2 -> 30.4%) and L2 throughput drops (38.9 -> 25.6%): per-thread register reuse produces the same output with less aggregate memory demand per cycle.
- There is speedup present. Lets confirm the metrics that we assumed to increase - derived__avg_thread_executed,launch__registers_per_thread

void <unnamed>::multi_tile_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 4, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                      Metric Unit         Metric Value
  ---------------------------- --------------- --------------------
  derived__avg_thread_executed          thread              219,232
  launch__registers_per_thread register/thread                  255


---

## 4. warp_rebalance - [04_warp_rebalance.cu](../04_warp_rebalance.cu)

Same 64x64 SMEM tile as regblock. Block grows from 16x4 = 64 threads to
16x8 = 128 threads, register tile shrinks from 16x4 to 8x4 (64 -> 32
outputs / thread), and `__launch_bounds__(128, 3)` pins 3 resident blocks
per SM. The trade is: half the per-thread reuse regblock won, in exchange
for twice the resident warps to feed the schedulers. From the code you
can see the inner loop is otherwise identical to regblock - the change
is a reshape of work across threads.

**Speed of Light Throughput Summary**

void <unnamed>::multi_tile_v2_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.99
    SM Frequency                    Ghz         1.24
    Elapsed Cycles                cycle    9,456,238
    Memory Throughput                 %        89.25
    DRAM Throughput                   %        36.94
    Duration                         ms         7.64
    L1/TEX Cache Throughput           %        89.21
    L2 Cache Throughput               %        31.37
    SM Active Cycles              cycle 9,423,035.06
    Compute (SM) Throughput           %        89.25
    ----------------------- ----------- ------------

- Memory and compute SOL climb back to ~89% (from regblock's ~55%). Both rise together, so the SM is no longer sitting idle waiting on stalls - this is the latency-hiding effect of more resident warps.
- Elapsed cycles drop ~18% from regblock (11.56M -> 9.46M). This is the wall-time win, and it is from cycles being filled, not from any reduction in the work itself.
- L1/TEX cache throughput at 89.2% looks similar to early kernels, but the meaning is different - here it is SMEM traffic from 12 active warps per SM, not the LSU bottleneck seen before.
- As the main change is regarding the occupancy, lets look at Occupany and Warp State Sections.

**Occupancy and Warp State**

void <unnamed>::multi_tile_v2_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle         4.20
    ---------------------------------------- ----------- ------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            3
    Block Limit Shared Mem                block            3
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           12
    Theoretical Occupancy                     %           25
    Achieved Occupancy                        %        24.01
    Achieved Active Warps Per SM           warp        11.53
    ------------------------------- ----------- ------------

- Block Limit Registers = 3 and Block Limit Shared Mem = 3 - both limits at 3 blocks/SM. Loosening just one would not unlock a fourth block.
- Theoretical active warps doubles 6 -> 12 vs regblock; achieved occupancy is 24.0% (regblock was ~12%).
- Warp cycles per issued instruction = 4.2, with the dominant stall reason being "Not Selected" at 30.7% (1.3 of 4.2 cycles). This means eligible warps exist every cycle, the scheduler is choosing among them rather than waiting on a memory dependency or pipeline bubble.

**Metrics to confirm warp parallelism is the lever**

Metrics used:
- sm__issue_active.avg.pct_of_peak_sustained_elapsed
- sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
- sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
- smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
- launch__registers_per_thread

  void <unnamed>::multi_tile_v2_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                                                               Metric Unit         Metric Value
  --------------------------------------------------------------------- --------------- --------------------
  launch__registers_per_thread                                          register/thread                  168
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active                         %                89.30
  sm__issue_active.avg.pct_of_peak_sustained_elapsed                                  %                68.40
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed                        %                48.63
  smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio            inst                 1.29

- Issue % rises 53.5 -> 68.4 (+14.9). Here with twice the resident warps, the schedulers find something to issue more often.
- FMA tracks issue: 39.7 -> 48.6 (+8.9). The cycles regblock opened up are now being filled with FFMAs.
- LSU climbs back to 89.3%. Per-thread inner-loop SMEM traffic is unchanged from regblock - we are just doing more of it per cycle. LSU is the next thing to attack.
- regs / thread tightens 255 -> 168 (the smaller register tile fits in fewer regs), and that is what made room for the second resident block.

From these the next optimization is the LSU pipe at 89.3% - the inner-loop SMEM reads are now what fills cycles and what the next optimization targets.

## 5. bank_pad_vec - [05_bank_pad_vec.cu](../05_bank_pad_vec.cu)

Same 64x64 tile, 128 threads / block, 3 resident blocks / SM as
warp_rebalance. Two SMEM-side changes in the code: tileA stride goes
64 -> 65 floats (break an 8-row LDS bank conflict pattern), and
the inner-loop tileB load is reissued as a single LDS.128 - one float4
fetch in place of 4 scalar LDS instructions per thread per k-step. The
per-thread column layout also flips from {tx, tx+16, tx+32, tx+48} to
{tx*4, tx*4+1, tx*4+2, tx*4+3} so the float4 sits 16B-aligned.

**Speed of Light Throughput Summary**

void <unnamed>::bank_pad_vec_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.99
    SM Frequency                    Ghz         1.24
    Elapsed Cycles                cycle    8,275,015
    Memory Throughput                 %        65.31
    DRAM Throughput                   %        42.26
    Duration                         ms         6.69
    L1/TEX Cache Throughput           %        65.48
    L2 Cache Throughput               %        36.04
    SM Active Cycles              cycle 8,256,677.31
    Compute (SM) Throughput           %        74.69
    ----------------------- ----------- ------------

- First kernel where compute SOL leads memory SOL: 74.7% compute vs 65.3% memory.
- Memory SOL drops 89.3 -> 65.3% by design - LDS.128 is one instruction for what used to be four, so the LSU pipe issues fewer SMEM-read instructions per cycle.
- DRAM throughput climbs 36.9 -> 42.3% even while memory SOL falls. The byte volume is roughly unchanged; the kernel just finishes faster (7.64 ms -> 6.69 ms), so bytes/second goes up.

**Memory Workload Analysis**

void <unnamed>::bank_pad_vec_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Memory Workload Analysis
    -------------------------------------- ----------- ------------
    Metric Name                            Metric Unit Metric Value
    -------------------------------------- ----------- ------------
    Memory Throughput                          Gbyte/s        80.98
    Mem Busy                                         %        65.31
    Max Bandwidth                                    %        63.94
    L1/TEX Hit Rate                                  %         4.11
    L2 Hit Rate                                      %        52.17
    Mem Pipes Busy                                   %        63.94
    -------------------------------------- ----------- ------------

- The Memory Workload Tables mentions two follow-on inefficiencies that this kernel does not address:
  - Shared-store bank conflicts: 1.1-way avg over 8.39M shared store requests -> 1.23M conflicts, 12.73% of shared-store wavefronts (Est. Speedup 8.3%).
  - Global-store sector utilization: only 8 of 32 bytes per sector are used by each thread on the C-matrix store (Est. Speedup ~49%). The narrow C-write pattern is the same one tiled had - the LDS.128 usage is on the load side, the store side is left alone, adding this might improve performance.

**Metrics to confirm the issue**

Metrics used:
- sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
- sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
- sm__issue_active.avg.pct_of_peak_sustained_elapsed
- l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
- l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

  void <unnamed>::bank_pad_vec_kernel<2>(const float *, const float *, float *, int, int, int) (16, 16, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                                                               Metric Unit         Metric Value
  --------------------------------------------------------------------- --------------- --------------------
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                            37,222
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                         1,233,887
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active                         %                64.13
  sm__issue_active.avg.pct_of_peak_sustained_elapsed                                  %                74.80
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed                        %                56.31

- The pad+1 was nominally targeted at SMEM-load bank conflicts, but the load-side conflict count actually *increases* 16,966 -> 37,222. The new inner-loop access geometry (different per-thread column layout + LDS.128 on tileB + stride-65 tileA) ends up introducing more conflicts than it removed. SMEM-store conflicts barely move (1.25M -> 1.23M).
- The main advantage was actually in the instruction-count reduction at the LSU pipe (LDS.128 on the b-load), not the bank-conflict reduction at the SMEM banks. This gives one important factor in GPU kernel optimization: the change that "should have" helped on paper is not always the change carrying the speedup.

Improving this kernel and further analysis are future work.

---

## - cublas (reference) - cuBLAS `gemmEx`, `CUBLAS_COMPUTE_32F`

Included as the reference ceiling on the
same hardware and shape, routed through `cublasGemmEx` with
`CUBLAS_GEMM_DEFAULT` (non-Tensor-Core) as the our GEMM was CUDA core based.
The kernel that actually runs is
`ampere_sgemm_128x128_nn`. There is no code to read for this row, so we
go straight to the profile.

**Speed of Light Throughput Summary**

ampere_sgemm_128x128_nn (16, 16, 2)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.99
    SM Frequency                    Ghz         1.24
    Elapsed Cycles                cycle    6,511,085
    Memory Throughput                 %        50.11
    DRAM Throughput                   %        18.46
    Duration                         ms         5.26
    L1/TEX Cache Throughput           %        50.14
    L2 Cache Throughput               %        24.56
    SM Active Cycles              cycle 6,507,643.31
    Compute (SM) Throughput           %        74.17
    ----------------------- ----------- ------------

- Compute SOL 74.2% - similar to bank_pad_vec at 74.7%. cuBLAS does not pull more out of the FMA pipe in % terms; it just does it in less wall time.
- Memory SOL 50.1% (vs our 65.3%) and DRAM throughput 18.5% (vs our 42.3%). Same arithmetic, much less DRAM bandwidth - what we are missing is cache reuse across blocks.

**Metrics to note**

Metrics used:
- l1tex__t_sector_hit_rate.pct
- lts__t_sector_hit_rate.pct
- sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed
- sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
- launch__registers_per_thread

  ampere_sgemm_128x128_nn (16, 16, 2)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
  Metric Name                                                      Metric Unit         Metric Value
  ------------------------------------------------------------ --------------- --------------------
  l1tex__t_sector_hit_rate.pct                                               %                 0.03
  launch__registers_per_thread                                 register/thread                  118
  lts__t_sector_hit_rate.pct                                                 %                82.28
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active                %                50.14
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed               %                64.53

| metric | bank_pad_vec | cublas | delta |
|---|---|---|---|
| L2 hit rate | 52.2% | 82.3% | +30.1 pp |
| FMA pipe util | 56.3% | 64.5% | +8.2 pp |
| LSU pipe util | 64.1% | 50.1% | -14.0 pp |
| achieved occupancy | 23.9% | 32.9% | +9.0 pp |
| regs / thread | 168 | 118 | -50 |

- L2 hit rate is the main difference (+30). cuBLAS orders block launches so each new block's input tiles tend to already sit in L2 from prior blocks sharing a row or column - an L2-level reuse layer we do not implement. Our reuse pattern optimization sits only at the register and L1/TEX level.
- 50 fewer regs / thread leads to the higher achieved occupancy (32.9% vs our 23.9%), which buys higher FMA %.
- The 1.37x gap (5.92 ms -> 4.31 ms) is not a property of any per-block technique tried in rows 0-5. Closing it requires cross-block scheduling / L2 tile-swizzle, which sits outside the per-kernel optimizations covered here.

One more thing to note is that this document lets you optimize the speedup fairly quickly, but for a single shape, not all of them. If we consider tall, short,  skinny or wide matrices. The same metrics are not affected. The workload imbalance becomes a major issue. We would then need to apply additional heuristics to get the best performance for various sizes. CUTLASS/cuBLASLt is a better tool to use in that case for initial development, you have a little more freedom than cuBLAS on the various parameters.
---