#!/bin/bash
# Submit 8 identical repeat runs of ONE fixed configuration (depth 8, matrix LR 0.025) to check how
# reproducible the wall-clock timing is. The runs differ only in their name (rep1..rep8).
# Default config = the old sweep config with MuonEq, Muon+ and NorMuon ON (variant "eq").
#
#   ./launch_repeat.sh                       # submit repetitions 1..8
#   DRY_RUN=1 ./launch_repeat.sh             # print the sbatch commands only
#   REPS="1 2 3" ./launch_repeat.sh          # subset
#   VARIANT=base POSTORTH=off ./launch_repeat.sh   # a different fixed config (new tags, no clash)
#
# Everything lives under repeat_* names / logs/slurm/repeat / logs/repeat, so the old sweep_* and
# plain_* logs, .done markers and checkpoints are never touched. Finished runs are skipped by
# repeat.sbatch, so re-running only redoes missing repetitions (but don't re-run while jobs are still
# queued/running: the .done marker only appears when a run finishes).
#
# Time limit / GPU / DEV_BS are the d8 values of the old launcher. Keep the GPU type fixed,
# otherwise wall-clock comparisons between the repeats are contaminated.

set -euo pipefail
cd "$(dirname "$0")"

REPS="${REPS:-1 2 3 4 5 6 7 8}"
VARIANT="${VARIANT:-eq}"
POSTORTH="${POSTORTH:-on}"
DEPTH=8
MATRIX_LR=0.025
PARTITION="${PARTITION:-gpu-long}"
TIME_LIMIT="01:00:00"
GPU="h100-47:1"
DEV_BS=32

n=0
for rep in $REPS; do
    tag="repeat_d${DEPTH}_${VARIANT}_po${POSTORTH}_lr${MATRIX_LR}_rep${rep}"
    cmd=(sbatch
        --job-name="$tag"
        --partition="$PARTITION"
        --time="$TIME_LIMIT"
        --gpus="$GPU"
        --export="ALL,REP=$rep,VARIANT=$VARIANT,POSTORTH=$POSTORTH,DEPTH=$DEPTH,MATRIX_LR=$MATRIX_LR,DEV_BS=$DEV_BS"
        repeat.sbatch)
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "${cmd[*]}"
    else
        "${cmd[@]}"
    fi
    n=$((n + 1))
done
echo "Submitted $n jobs"
