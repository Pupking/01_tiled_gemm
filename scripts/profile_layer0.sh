#!/usr/bin/env bash
# Layer-0 profile capture. One .ncu-rep per kernel at 2048^3.
# Low iters/runs because ncu replays each kernel ~15x for the full metric set.

set -euo pipefail

cd "$(dirname "$0")/.."

BENCH=./build/bin/gemm_bench
OUTDIR=profiles
mkdir -p "$OUTDIR"

KERNELS=(naive tiled tiled_coalesced regblock warp_rebalance
         bank_pad_vec cublas)

for k in "${KERNELS[@]}"; do
    echo ">>> profiling $k"
    ncu --set full \
        --force-overwrite \
        --export "$OUTDIR/$k.ncu-rep" \
        "$BENCH" --M 2048 --N 2048 --K 2048 \
                 --iters 1 --warmup 0 --runs 1 \
                 --kernel "$k"
done

echo "done. profiles in $OUTDIR/"
ls -lh "$OUTDIR/"
