#!/bin/bash
#SBATCH --job-name=nanochat-timing
#SBATCH --partition=gpu-long
#SBATCH --time=03:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

# Estimate how long a FULL run takes for several model sizes, without running them.
#
# For each depth, this runs the REAL configuration (same auto batch size, same --target-param-data-ratio,
# same planned num_iterations) but exits after STEPS steps (--stop-after-steps). The steady-state step time is then
# extrapolated to the full num_iterations, together with the cost of the periodic val-bpb / CORE evals and
# the fixed startup + torch.compile overhead. See summarize_timing.py for the exact model.
#
# Why a few dozen steps instead of 10-20% of the run: step time is essentially constant (fixed shapes, fixed batch),
# so ~60 timed steps pins it down to about a percent. Running 10-20% would cost hours for the big models for no gain.
#
# Usage (put timing.sbatch and summarize_timing.py in the repo root; submit from there):
#   sbatch timing.sbatch
#   sbatch --export=ALL,DEPTHS="12 16 20 24" timing.sbatch
#
# Optional overrides via --export=ALL,KEY=value,...
#   DEPTHS     depths to test                              (default "8 10 12 14 16 18 20")
#   VARIANT    eq | eq_m2 | m2 | base  (MuonEq / Muon^2)   (default eq = standard nanochat)
#   STEPS      steps per timing run, must be even          (default 80; first 20 are ignored when averaging)
#   RATIO      --target-param-data-ratio                    (default 12)
#   DEVICE_BS  starting --device-batch-size                 (default 32; halved automatically on OOM)
#   MIN_DEVICE_BS  give up below this                       (default 4)
#   FULL_EVAL_EVERY / FULL_CORE_EVERY  eval cadence of the REAL run, used for the extrapolation (250 / 2000)
#   MEASURE_CORE  1 = also time the CORE eval; 0 = skip it (use 0 if you run with --core-metric-every=-1,
#                 or if CORE data is unavailable on the compute node)                    (default 1)
#   MUON2_BETA2, MUON2_EPS                                  (0.95 / 1e-8)
#
# Caveat: timings are for the GPU type of THIS job. Run your real jobs on the same hardware (the DEVICE_BS that
# survived, i.e. the "dev_bs" column, is also the largest one that fits in memory on it).

set -euo pipefail

DEPTHS="${DEPTHS:-8 10 12 14 16 18 20}"
VARIANT="${VARIANT:-eq}"
STEPS="${STEPS:-80}"
RATIO="${RATIO:-12}"
DEVICE_BS="${DEVICE_BS:-32}"
MIN_DEVICE_BS="${MIN_DEVICE_BS:-4}"
FULL_EVAL_EVERY="${FULL_EVAL_EVERY:-250}"
FULL_CORE_EVERY="${FULL_CORE_EVERY:-2000}"
MEASURE_CORE="${MEASURE_CORE:-1}"
MUON2_BETA2="${MUON2_BETA2:-0.95}"
MUON2_EPS="${MUON2_EPS:-1e-8}"

if (( STEPS % 2 != 0 || STEPS < 40 )); then echo "STEPS must be even and >= 40 (got $STEPS)" >&2; exit 1; fi
HALF=$(( STEPS / 2 ))   # evals at steps 0, HALF, STEPS -> the later ones are free of compile cost
if [[ "$MEASURE_CORE" == "1" ]]; then CORE_EVERY="$HALF"; else CORE_EVERY=-1; FULL_CORE_EVERY=-1; fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
RUN_DIR="$REPO_ROOT/logs/timing/${SLURM_JOB_ID:-$$}_${VARIANT}"
mkdir -p "$LOG_DIR" "$RUN_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${VARIANT}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1

M2_ARGS=(--muon2 --muon2-beta2="$MUON2_BETA2" --muon2-eps="$MUON2_EPS")
case "$VARIANT" in
    eq)    VARIANT_ARGS=(--muoneq) ;;
    eq_m2) VARIANT_ARGS=(--muoneq "${M2_ARGS[@]}") ;;
    m2)    VARIANT_ARGS=(--no-muoneq "${M2_ARGS[@]}") ;;
    base)  VARIANT_ARGS=(--no-muoneq) ;;
    *) echo "Unknown VARIANT='$VARIANT' (expected: eq | eq_m2 | m2 | base)" >&2; exit 1 ;;
esac

echo "Timing job: variant=$VARIANT depths=[$DEPTHS] steps=$STEPS ratio=$RATIO start device_bs=$DEVICE_BS"
echo "Per-depth logs in: $RUN_DIR"

for depth in $DEPTHS; do
    dbs="$DEVICE_BS"
    while true; do
        log="$RUN_DIR/d${depth}.log"
        echo "=== depth=$depth device_bs=$dbs ==="
        start=$(date +%s)
        rc=0
        python -m scripts.base_train \
            --depth="$depth" \
            --run=dummy \
            --model-tag="timing_d${depth}" \
            --target-param-data-ratio="$RATIO" \
            --device-batch-size="$dbs" \
            --stop-after-steps="$STEPS" \
            --eval-every="$HALF" \
            --core-metric-every="$CORE_EVERY" \
            --sample-every=-1 \
            --save-every=-1 \
            "${VARIANT_ARGS[@]}" 2>&1 | tee "$log" || rc=$?
        wall=$(( $(date +%s) - start ))

        status="ok"
        if (( rc != 0 )); then
            if grep -q -i "out of memory" "$log"; then
                status="oom"
            else
                status="failed"
            fi
        fi
        printf 'wall_s=%s\nrc=%s\ndevice_bs=%s\nstatus=%s\n' "$wall" "$rc" "$dbs" "$status" > "$RUN_DIR/d${depth}.meta"

        if [[ "$status" == "oom" ]] && (( dbs / 2 >= MIN_DEVICE_BS )); then
            dbs=$(( dbs / 2 ))
            echo "OOM at depth=$depth -> retrying with device_bs=$dbs"
            continue
        fi
        break
    done
    echo "depth=$depth finished: status=$status wall=${wall}s device_bs=$dbs"
done

echo
if [[ -f "$REPO_ROOT/summarize_timing.py" ]]; then
    python "$REPO_ROOT/summarize_timing.py" "$RUN_DIR" \
        --eval-every "$FULL_EVAL_EVERY" --core-every "$FULL_CORE_EVERY" | tee "$RUN_DIR/summary.txt"
else
    echo "summarize_timing.py not found in $REPO_ROOT; run it manually on $RUN_DIR"
fi
