#!/bin/bash
#SBATCH --job-name=nanochat-sweep
#SBATCH --partition=gpu-long
#SBATCH --time=08:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

# One (VARIANT, DEPTH, MATRIX_LR) run per job. Normally submitted by launch_sweep.sh,
# which overrides --time / --gpus / --job-name per depth on the sbatch command line.
#
# Required (--export=ALL,KEY=value,...):
#   VARIANT    eq | eq_m2 | m2 | base
#   DEPTH      model depth (8, 12, 16)
#   MATRIX_LR  value passed to --matrix-lr
#
# Optional:
#   DEV_BS (32)   per-device batch size; lower it (16, 8) if you OOM. Total batch size is
#                 unchanged (auto-computed by nanochat), the run just uses more grad-accum steps.
#   EVAL_EVERY / EVAL_TOKENS / CORE_EVERY   only passed if set; otherwise nanochat defaults
#   MUON2_BETA2 (0.95)  MUON2_EPS (1e-8)
#
# Everything else (training horizon via target-param-data-ratio, auto batch size, weight decay,
# embedding/unembedding/scalar LRs, warmup/warmdown) is left at nanochat defaults.
#
# Re-submitting a finished run is a no-op (skipped via logs/sweep/<tag>.done).

set -euo pipefail

VARIANT="${VARIANT:?set VARIANT=eq|eq_m2|m2|base}"
DEPTH="${DEPTH:?set DEPTH}"
MATRIX_LR="${MATRIX_LR:?set MATRIX_LR}"
DEV_BS="${DEV_BS:-32}"
MUON2_BETA2="${MUON2_BETA2:-0.95}"
MUON2_EPS="${MUON2_EPS:-1e-8}"

TAG="sweep_d${DEPTH}_${VARIANT}_lr${MATRIX_LR}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm/sweep"
DONE_DIR="$REPO_ROOT/logs/sweep"
mkdir -p "$LOG_DIR" "$DONE_DIR"

if [[ -f "$DONE_DIR/$TAG.done" ]]; then
    echo "[$TAG] already finished, skipping"
    exit 0
fi

exec > "$LOG_DIR/${TAG}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1

ARGS=(
    --depth="$DEPTH"
    --run=dummy
    --model-tag="$TAG"
    --matrix-lr="$MATRIX_LR"
    --device-batch-size="$DEV_BS"
    --sample-every=-1              # text samples are not needed for the sweep
)

[[ -n "${EVAL_EVERY:-}"  ]] && ARGS+=(--eval-every="$EVAL_EVERY")
[[ -n "${EVAL_TOKENS:-}" ]] && ARGS+=(--eval-tokens="$EVAL_TOKENS")
[[ -n "${CORE_EVERY:-}"  ]] && ARGS+=(--core-metric-every="$CORE_EVERY")

M2_ARGS=(--muon2 --muon2-beta2="$MUON2_BETA2" --muon2-eps="$MUON2_EPS")
case "$VARIANT" in
    eq)    ARGS+=(--muoneq) ;;
    eq_m2) ARGS+=(--muoneq "${M2_ARGS[@]}") ;;
    m2)    ARGS+=(--no-muoneq "${M2_ARGS[@]}") ;;
    base)  ARGS+=(--no-muoneq) ;;
    *) echo "Unknown VARIANT='$VARIANT' (expected: eq | eq_m2 | m2 | base)" >&2; exit 1 ;;
esac

echo "[$TAG] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[$TAG] python -m scripts.base_train ${ARGS[*]}"
python -m scripts.base_train "${ARGS[@]}"

# Only reached if training exited cleanly (set -e)
date -Is > "$DONE_DIR/$TAG.done"
