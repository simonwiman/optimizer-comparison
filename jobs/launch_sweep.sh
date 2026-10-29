#!/bin/bash
# Submit the full sweep: 3 depths x 5 matrix LRs x 4 variants = 60 independent jobs.
#
#   ./launch_sweep.sh                 # submit everything
#   DRY_RUN=1 ./launch_sweep.sh       # print the sbatch commands only
#   DEPTHS="8" ./launch_sweep.sh      # subset (also: LRS="0.02 0.04", VARIANTS="eq base")
#
# Finished runs are skipped by sweep.sbatch, so re-running after a failure only redoes
# the missing ones (completed jobs exit immediately).
#
# Per-depth wall-clock limits are ~2x the timing-table TOTAL_h (0.29 / 1.26 / 4.41 h).
# Per-depth GPU type: d16 peaked at 47.1 GiB with dev_bs=32 in the timing job, which is
# too tight for a ~47 GB card, so d16 defaults to h100-96. Keep the GPU type fixed within a
# depth, otherwise wall-clock comparisons between variants are contaminated.

set -euo pipefail
cd "$(dirname "$0")"

DEPTHS="${DEPTHS:-16 12 8}"          # longest first so they start early
LRS="${LRS:-0.005 0.01 0.02 0.04 0.08}"
VARIANTS="${VARIANTS:-eq eq_m2 m2 base}"
PARTITION="${PARTITION:-gpu-long}"

declare -A TIME_LIMIT=( [8]=01:00:00 [12]=03:00:00 [16]=09:00:00 )
declare -A GPU=(        [8]=h100-47:1 [12]=h100-47:1 [16]=h100-96:1 )
declare -A DEVBS=(      [8]=32 [12]=32 [16]=32 )

n=0
for depth in $DEPTHS; do
    for lr in $LRS; do
        for v in $VARIANTS; do
            tag="sweep_d${depth}_${v}_lr${lr}"
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
