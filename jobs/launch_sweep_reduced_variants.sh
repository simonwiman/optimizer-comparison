#!/bin/bash
# Submit the sweep: 2 depths x 3 matrix LRs x 4 variants = 24 independent jobs.
# Muon+ and NorMuon are OFF in every job (set in sweep.sbatch); the variants differ only in
# MuonEq / Muon^2:
#   base   : no MuonEq, no Muon^2
#   eq     : MuonEq
#   m2     : Muon^2
#   eq_m2  : MuonEq + Muon^2
#
#   ./launch_sweep.sh                 # submit everything
#   DRY_RUN=1 ./launch_sweep.sh       # print the sbatch commands only
#   DEPTHS="8" ./launch_sweep.sh      # subset (also: LRS="0.02 0.04", VARIANTS="eq base")
#
# Finished runs are skipped by sweep.sbatch, so re-running after a failure only redoes
# the missing ones (completed jobs exit immediately).
#
# Time limits and GPU types are carried over from the previous sweep (d8 / d12 are the
# same models, and Muon+/NorMuon are cheap, so the old limits are still ~2x what is needed).
# Keep the GPU type fixed within a depth, otherwise wall-clock comparisons between
# variants are contaminated.

set -euo pipefail
cd "$(dirname "$0")"

DEPTHS="${DEPTHS:-12 8}"             # longest first so they start early
LRS="${LRS:-0.01 0.02 0.04}"
VARIANTS="${VARIANTS:-eq eq_m2 m2 base}"
PARTITION="${PARTITION:-gpu-long}"

declare -A TIME_LIMIT=( [8]=01:00:00 [12]=03:00:00 )
declare -A GPU=(        [8]=h100-47:1 [12]=h100-47:1 )
declare -A DEVBS=(      [8]=32 [12]=32 )

n=0
for depth in $DEPTHS; do
    if [[ -z "${TIME_LIMIT[$depth]:-}" ]]; then
        echo "No time limit / GPU configured for depth $depth (add it to the arrays)" >&2
        exit 1
    fi
    for lr in $LRS; do
        for v in $VARIANTS; do
            tag="plain_d${depth}_${v}_lr${lr}"
            cmd=(sbatch
                --job-name="$tag"
                --partition="$PARTITION"
                --time="${TIME_LIMIT[$depth]}"
                --gpus="${GPU[$depth]}"
                --export="ALL,VARIANT=$v,DEPTH=$depth,MATRIX_LR=$lr,DEV_BS=${DEVBS[$depth]}"
                sweep.sbatch)
            if [[ -n "${DRY_RUN:-}" ]]; then
                echo "${cmd[*]}"
            else
                "${cmd[@]}"
            fi
            n=$((n + 1))
        done
    done
done
echo "Submitted $n jobs"
