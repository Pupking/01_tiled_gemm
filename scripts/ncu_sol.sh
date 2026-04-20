#!/usr/bin/env bash
# Print SpeedOfLight section from a layer-0 profile.
# Usage: scripts/ncu_sol.sh <name>     e.g. naive, tiled, regblock

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <name>" >&2
    echo "names: naive tiled tiled_coalesced regblock warp_rebalance bank_pad_vec cublas" >&2
    exit 1
fi

NAME=$1
REP=profiles/${NAME}.ncu-rep

case "$NAME" in
    naive)            KERNEL=naive_gemm_kernel ;;
    tiled)            KERNEL=tiled_gemm_kernel ;;
    tiled_coalesced)  KERNEL=tiled_coalesced_gemm_kernel ;;
    regblock)         KERNEL=multi_tile_kernel ;;
    warp_rebalance)   KERNEL=multi_tile_v2_kernel ;;
    bank_pad_vec)     KERNEL=bank_pad_vec_kernel ;;
    cublas)           KERNEL=sgemm ;;
    *) echo "unknown name: $NAME" >&2; exit 1 ;;
esac

[[ -f "$REP" ]] || { echo "missing profile: $REP" >&2; exit 1; }

ncu --import "$REP" --print-details header --section SpeedOfLight -k "$KERNEL"
